import logging
from argparse import Namespace
from copy import deepcopy
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import List

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from bond.data.batching import BatchedExamples
from bond.data.dataset import DatasetName, DatasetType, load_transformed_dataset, load_dataset, load_tags_dict, SubTokenDataset
from bond.utils import Scores, initialize_roberta, ner_scores, soft_frequency, convert_hard_to_soft_labels

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class TrainingFramework(Enum):
    BOND = 'bond'
    SUPERVISED = 'supervised'


def evaluate(args: Namespace, model: PreTrainedModel, eval_dataset: SubTokenDataset, dataset_name: DatasetName) -> Scores:
    eval_sampler = SequentialSampler(eval_dataset)
    batch_size = args.batch_size * 2
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size, collate_fn=eval_dataset.collate_fn)

    eval_loss = 0.0
    nb_eval_steps = 0
    predicted_labels: List[int] = []
    true_labels: List[int] = []

    model.eval()
    for batch in tqdm(eval_dataloader, desc=f"Evaluating", leave=False):
        batch: BatchedExamples
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(batch)

            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1

        batch_raveled_logits = torch.cat(list(logits.detach().cpu()), dim=0)
        batch_raveled_main_sentences_mask = torch.cat(list(batch.main_sentences_mask.cpu()), dim=0)
        batch_raveled_true_labels = torch.cat(list(batch.label_ids.cpu()), dim=0)
        batch_predicted_labels = torch.argmax(batch_raveled_logits, dim=-1).masked_select(batch_raveled_main_sentences_mask)
        batch_true_labels = batch_raveled_true_labels.masked_select(batch_raveled_main_sentences_mask)

        assert len(batch_predicted_labels) == len(batch_true_labels)

        predicted_labels.extend(batch_predicted_labels.tolist())
        true_labels.extend(batch_true_labels.tolist())

    eval_loss = eval_loss / nb_eval_steps
    results = ner_scores(true_labels, predicted_labels, load_tags_dict(dataset_name))
    results['loss'] = eval_loss

    return results


def prepare_dataset(args: Namespace, dataset: DatasetName, dataset_type: DatasetType, tokenizer: PreTrainedTokenizer) -> SubTokenDataset:
    if dataset_type == DatasetType.DISTANT:
        ds = load_transformed_dataset(
            dataset, args.add_gold_labels, tokenizer, args.model_name,
            max_seq_length=args.max_seq_length,
            base_distributions_file=args.base_distributions_file,
            add_distant=args.add_distant
        )
    else:
        ds = load_dataset(dataset, dataset_type, tokenizer, args.model_name, args.max_seq_length)

    return ds


def train_bond(
        args: Namespace,
        model: PreTrainedModel,
        dataset_name: DatasetName,
        train_dataset: SubTokenDataset,
        eval_dataset: SubTokenDataset,
        tb_writer: SummaryWriter,
        amp_scaler: torch.cuda.amp.GradScaler
):
    """Train model for ner_fit_epochs epochs then do self training for self_training_epochs epochs"""

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    num_labels = len(load_tags_dict(dataset_name).keys())

    gradient_accumulation_steps = args.gradient_accumulation_steps

    min_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    max_steps_per_epoch = min_steps_per_epoch + 1

    st_epochs = args.self_training_epochs
    st_steps = st_epochs * max_steps_per_epoch

    ner_epochs = args.ner_fit_epochs
    ner_steps = ner_epochs * max_steps_per_epoch

    total_steps = ner_steps
    warmup_steps = int(args.warmup_proportion * max_steps_per_epoch)
    warmup_batches = warmup_steps * gradient_accumulation_steps

    # prepare scheduler for NER fitting stage
    model, optimizer, scheduler = initialize_roberta(
        args, model, total_steps,
        warmup_steps=warmup_steps,
        end_lr_proportion=args.self_training_lr_proportion
    )

    # Train!

    global_batch = 0
    examples_from_last_log = 0
    steps_from_last_log = 0

    tr_loss, logging_loss = 0.0, 0.0

    def log_metrics(res: Scores, prefix: str) -> None:
        examples_seen = global_batch * args.batch_size

        print('Logging metrics:')
        pprint(res)

        for metric_name, metric_value in res.items():
            tb_writer.add_scalar(f"{metric_name}_{prefix}", metric_value, examples_seen)
            tb_writer.add_scalar(f"{metric_name}", metric_value, examples_seen)

        for group_idx, group in enumerate(optimizer.param_groups):
            tb_writer.add_scalar(f"lr_{group.get('name', f'group{group_idx}')}", group['lr'], examples_seen)

    # Fitting BERT to NER task

    # warmup epoch
    epoch_iterator = tqdm(train_dataloader, desc=f'Warmup epoch!', total=warmup_batches)
    for batch_idx, batch in enumerate(epoch_iterator):
        if batch_idx >= warmup_batches:
            epoch_iterator.close()
            continue

        batch: BatchedExamples
        model.train()
        with torch.cuda.amp.autocast():
            outputs = model(batch, self_training=False, warmup=True, use_kldiv_loss=args.use_kldiv_loss_ner)
        loss, logits = outputs[0], outputs[1]  # model outputs are always tuple in pytorch-transformers
        loss = loss / gradient_accumulation_steps

        amp_scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            amp_scaler.step(optimizer)
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()

            amp_scaler.update()

    patience = args.adaptive_scheduler_patience
    best_result = 0.0
    best_model = deepcopy(model)

    curr_lr = args.learning_rate * args.self_training_lr_proportion
    max_lr = optimizer.param_groups[0]['lr']
    for group in optimizer.param_groups:
        group['lr'] = (group['lr'] / max_lr) * curr_lr  # to keep lr layer decay

    for ner_fit_epoch in range(ner_epochs):

        total_batches = len(train_dataloader)

        epoch_iterator = tqdm(train_dataloader, desc=f'Fitting NER on {ner_fit_epoch + 1}/{ner_epochs} epoch', total=total_batches)
        for batch_idx, batch in enumerate(epoch_iterator):
            global_batch += 1
            examples_from_last_log += args.batch_size

            batch: BatchedExamples
            model.train()
            with torch.cuda.amp.autocast():
                outputs = model(batch, self_training=False, use_kldiv_loss=args.use_kldiv_loss_ner)
            loss, logits = outputs[0], outputs[1]  # model outputs are always tuple in pytorch-transformers
            loss = loss / gradient_accumulation_steps

            amp_scaler.scale(loss).backward()

            tr_loss += loss.item()
            if global_batch % gradient_accumulation_steps == 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                amp_scaler.step(optimizer)
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                steps_from_last_log += 1

                amp_scaler.update()

            if examples_from_last_log >= args.logging:
                # Log metrics
                results = evaluate(args, model, eval_dataset, dataset_name)
                results = {k + '_dev': v for k, v in results.items()}

                curr_lr = optimizer.param_groups[0]['lr']
                if args.use_adaptive_scheduler:
                    curr_result = results['f1_dev']

                    if curr_result < best_result:
                        patience -= 1
                    else:
                        patience = args.adaptive_scheduler_patience
                        best_result = curr_result
                        best_model = deepcopy(model)

                    if patience < 0:  # update learning rate and model
                        curr_lr = curr_lr * args.adaptive_scheduler_drop
                        for group in optimizer.param_groups:
                            group['lr'] = group['lr'] * args.adaptive_scheduler_drop
                        model.load_state_dict(best_model.state_dict())
                        patience = args.adaptive_scheduler_patience

                log_metrics({**results, 'loss': (tr_loss - logging_loss) / steps_from_last_log, 'lr': curr_lr, 'patience': patience}, 'ner')
                logging_loss = tr_loss
                examples_from_last_log = 0
                steps_from_last_log = 0

        if curr_lr < 1e-7:
            print(f'Early stopping at epoch {ner_fit_epoch}!')
            break

    model.load_state_dict(best_model.state_dict())  # load best model from ner tuning stage
    patience = args.adaptive_scheduler_patience

    self_training_teacher_model = deepcopy(model)
    self_training_teacher_model.eval()

    total_steps = st_steps
    args.bert_learning_rate *= args.self_training_lr_proportion
    args.head_learning_rate *= args.self_training_lr_proportion

    st_batch = 0
    total_st_batches = st_epochs * len(train_dataloader)
    batches_since_update = 0

    def get_batches_until_update():
        completion = st_batch / total_st_batches
        update_rate = (1 - completion) * args.start_updates + completion * args.end_updates
        return len(train_dataloader) / update_rate

    # prepare scheduler for self training stage
    model, optimizer, scheduler = initialize_roberta(args, model, total_steps, warmup_steps=0, end_lr_proportion=0)
    model.prepare_for_self_training()

    curr_lr = optimizer.param_groups[0]['lr']
    # Self training
    for self_training_epoch in range(st_epochs):

        total_batches = len(train_dataloader)
        epoch_iterator = tqdm(train_dataloader, desc=f'Self training on {self_training_epoch + 1}/{st_epochs} epoch', total=total_batches)

        for batch_idx, batch in enumerate(epoch_iterator):

            if batch_idx >= total_batches:
                epoch_iterator.close()
                continue

            if batches_since_update > get_batches_until_update():
                logging.info(f'Model updateed on batch {st_batch}/{total_st_batches}')
                self_training_teacher_model = deepcopy(model)
                self_training_teacher_model.eval()
                batches_since_update = 0

            global_batch += 1
            st_batch += 1
            batches_since_update += 1
            examples_from_last_log += args.batch_size

            batch: BatchedExamples

            # Using current teacher to update the labels
            batch_without_labels = batch.without_labels()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = self_training_teacher_model(batch_without_labels)

            predictions = outputs[0]
            if args.correct_frequency:
                teacher_label_distributions = soft_frequency(logits=predictions, power=2, probs=True)
            else:
                teacher_label_distributions = predictions

            _threshold = args.label_keep_threshold
            teacher_mask = (teacher_label_distributions.max(dim=-1)[0] > _threshold)

            gold_label_mask = batch.gold_entities_mask.to(device)
            label_mask = batch.label_mask.to(device)
            label_ids = batch.label_ids.to(device)

            # correct distributions for known ground truth
            if not args.remove_guidance:
                teacher_label_distributions[gold_label_mask] = convert_hard_to_soft_labels(label_ids[gold_label_mask], num_labels)

            new_label_mask = (label_mask & teacher_mask) | gold_label_mask
            teacher_batch = batch.with_changes(label_distributions=teacher_label_distributions.cpu(), label_mask=new_label_mask.cpu())

            model.train()
            with torch.cuda.amp.autocast():
                outputs = model(teacher_batch, self_training=True, use_kldiv_loss=args.use_kldiv_loss)
            loss, logits = outputs[0], outputs[1]  # model outputs are always tuple in pytorch-transformers
            loss = loss / gradient_accumulation_steps

            amp_scaler.scale(loss).backward()

            tr_loss += loss.item()
            if global_batch % gradient_accumulation_steps == 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                amp_scaler.step(optimizer)
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                steps_from_last_log += 1

                amp_scaler.update()

            if examples_from_last_log >= args.logging:
                # Log metrics
                results = evaluate(args, model, eval_dataset, dataset_name)
                results = {k + '_dev': v for k, v in results.items()}

                curr_lr = optimizer.param_groups[0]['lr']
                if args.use_adaptive_scheduler:
                    curr_result = results['f1_dev']

                    if curr_result < best_result:
                        patience -= 1
                    else:
                        patience = args.adaptive_scheduler_patience
                        best_result = curr_result
                        best_model = deepcopy(model)

                    if patience < 0:  # update learning rate and model
                        curr_lr = curr_lr * args.adaptive_scheduler_drop
                        for group in optimizer.param_groups:
                            group['lr'] = group['lr'] * args.adaptive_scheduler_drop
                        model.load_state_dict(best_model.state_dict())
                        patience = args.adaptive_scheduler_patience

                log_metrics({**results, 'loss': (tr_loss - logging_loss) / steps_from_last_log, 'lr': curr_lr, 'patience': patience}, 'self_training')
                logging_loss = tr_loss
                examples_from_last_log = 0
                steps_from_last_log = 0

        if curr_lr < 1e-7:
            print(f'Early stopping at epoch {self_training_epoch}!')
            break

    tb_writer.close()

    return best_model


def train_supervised(
        args: Namespace,
        model: PreTrainedModel,
        dataset_name: DatasetName,
        train_dataset: SubTokenDataset,
        eval_dataset: SubTokenDataset,
        tb_writer: SummaryWriter,
        amp_scaler: torch.cuda.amp.GradScaler
):
    """Train model for ner_fit_epochs epochs"""

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)

    gradient_accumulation_steps = args.gradient_accumulation_steps

    min_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    max_steps_per_epoch = min_steps_per_epoch + 1

    ner_epochs = args.ner_fit_epochs
    ner_steps = ner_epochs * max_steps_per_epoch

    total_steps = ner_steps
    warmup_steps = int(args.warmup_proportion * max_steps_per_epoch)
    warmup_batches = warmup_steps * gradient_accumulation_steps

    model, optimizer, scheduler = initialize_roberta(args, model, total_steps, warmup_steps=warmup_steps)

    # Train!

    global_batch = 0
    steps_from_last_log = 0
    examples_from_last_log = 0

    tr_loss, logging_loss = 0.0, 0.0

    def log_metrics(res: Scores, prefix: str) -> None:
        examples_seen = global_batch * args.batch_size

        print('Logging metrics:')
        pprint(res)

        for metric_name, metric_value in res.items():
            tb_writer.add_scalar(f"{metric_name}_{prefix}", metric_value, examples_seen)
            tb_writer.add_scalar(f"{metric_name}", metric_value, examples_seen)

        for group_idx, group in enumerate(optimizer.param_groups):
            tb_writer.add_scalar(f"lr_{group.get('name', f'group{group_idx}')}", group['lr'], examples_seen)

    # Fitting BERT to NER task

    # warmup epoch
    epoch_iterator = tqdm(train_dataloader, desc=f'Warmup epoch!', total=warmup_batches)
    for batch_idx, batch in enumerate(epoch_iterator):
        if batch_idx >= warmup_batches:
            epoch_iterator.close()
            continue

        batch: BatchedExamples

        model.train()
        with torch.cuda.amp.autocast():
            outputs = model(batch, self_training=False, warmup=True, use_kldiv_loss=args.use_kldiv_loss_ner)
        loss, logits = outputs[0], outputs[1]  # model outputs are always tuple in pytorch-transformers
        loss = loss / gradient_accumulation_steps

        amp_scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            amp_scaler.step(optimizer)
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()

            amp_scaler.update()

    patience = args.adaptive_scheduler_patience
    best_result = 0.0
    best_model = deepcopy(model)

    curr_lr = optimizer.param_groups[0]['lr']
    for ner_fit_epoch in range(ner_epochs):

        total_batches = len(train_dataloader)

        epoch_iterator = tqdm(train_dataloader, desc=f'Fitting NER on {ner_fit_epoch + 1}/{ner_epochs} epoch', total=total_batches)
        for batch_idx, batch in enumerate(epoch_iterator):

            global_batch += 1
            examples_from_last_log += args.batch_size

            batch: BatchedExamples

            model.train()
            with torch.cuda.amp.autocast():
                outputs = model(batch, self_training=False, use_kldiv_loss=args.use_kldiv_loss_ner)
            loss, logits = outputs[0], outputs[1]  # model outputs are always tuple in pytorch-transformers
            loss = loss / gradient_accumulation_steps

            amp_scaler.scale(loss).backward()

            tr_loss += loss.item()
            if global_batch % gradient_accumulation_steps == 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                amp_scaler.step(optimizer)
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                steps_from_last_log += 1

                amp_scaler.update()

            if examples_from_last_log >= args.logging:
                # Log metrics
                results = evaluate(args, model, eval_dataset, dataset_name)
                results = {k + '_dev': v for k, v in results.items()}

                curr_lr = optimizer.param_groups[0]['lr']
                if args.use_adaptive_scheduler:
                    curr_result = results['f1_dev']

                    if curr_result < best_result:
                        patience -= 1
                    else:
                        patience = args.adaptive_scheduler_patience
                        best_result = curr_result
                        best_model = deepcopy(model)

                    if patience < 0:  # update learning rate and model
                        curr_lr = curr_lr * args.adaptive_scheduler_drop
                        for group in optimizer.param_groups:
                            group['lr'] = group['lr'] * args.adaptive_scheduler_drop
                        model.load_state_dict(best_model.state_dict())
                        patience = args.adaptive_scheduler_patience

                log_metrics({**results, 'loss': (tr_loss - logging_loss) / steps_from_last_log, 'lr': curr_lr, 'patience': patience}, 'ner')
                logging_loss = tr_loss
                examples_from_last_log = 0
                steps_from_last_log = 0

        if curr_lr < 1e-7:
            print(f'Early stopping at epoch {ner_fit_epoch}!')
            break

    return best_model


def train(
        args: Namespace,
        model: PreTrainedModel,
        dataset_name: DatasetName,
        train_dataset: SubTokenDataset,
        eval_dataset: SubTokenDataset,
        training_framework: TrainingFramework,
        tb_writer: SummaryWriter
):
    """Train model for ner_fit_epochs epochs then do self training for self_training_epochs epochs"""

    amp_scaler = torch.cuda.amp.GradScaler()

    if training_framework == TrainingFramework.BOND:
        return train_bond(args, model, dataset_name, train_dataset, eval_dataset, tb_writer, amp_scaler)
    elif training_framework == TrainingFramework.SUPERVISED:
        args.self_training_epochs = 0
        return train_supervised(args, model, dataset_name, train_dataset, eval_dataset, tb_writer, amp_scaler)
    else:
        raise ValueError(f'Unsupported training framework {training_framework.value}!')
