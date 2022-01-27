import math
from copy import deepcopy
from enum import Enum
from typing import List

import torch
from torch import softmax
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from bond.data import DatasetName, DatasetType, load_dataset, load_tags_dict
from bond.utils import Scores, initialize_roberta, ner_scores, soft_frequency

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class TrainingFramework(Enum):
    BOND = 'bond'
    NLL = 'nll'  # TODO
    SUPERVISED = 'supervised'


def evaluate(args, model: PreTrainedModel, dataset: DatasetName, dataset_type: DatasetType, tokenizer: PreTrainedTokenizer) -> Scores:
    eval_dataset = load_dataset(dataset, dataset_type, tokenizer, args.model_name, args.max_seq_length)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=eval_dataset.collate_fn)

    eval_loss = 0.0
    nb_eval_steps = 0
    predicted_labels: List[int] = []
    true_labels: List[int] = []

    model.eval()
    for batch in tqdm(eval_dataloader, desc=f"Evaluating on {dataset_type.value} dataset", leave=False):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            token_ids, token_mask, attention_mask, labels, label_mask = batch
            inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask,
                      "labels": labels, 'label_mask': label_mask}
            outputs = model(**inputs)

            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1

        batch_raveled_logits = torch.cat(list(logits.detach().cpu()), dim=0)
        batch_raveled_label_mask = torch.cat(list(label_mask.cpu()), dim=0)
        batch_raveled_true_labels = torch.cat(list(labels.cpu()), dim=0)
        batch_predicted_labels = torch.argmax(batch_raveled_logits, dim=-1).masked_select(batch_raveled_label_mask)
        batch_true_labels = batch_raveled_true_labels.masked_select(batch_raveled_label_mask)

        assert len(batch_predicted_labels) == len(batch_true_labels)

        predicted_labels.extend(batch_predicted_labels.tolist())
        true_labels.extend(batch_true_labels.tolist())

    eval_loss = eval_loss / nb_eval_steps
    results = ner_scores(true_labels, predicted_labels, load_tags_dict(dataset))
    results['loss'] = eval_loss

    return results


def train_bond(args, model: PreTrainedModel, dataset: DatasetName, dataset_type: DatasetType, tokenizer: PreTrainedTokenizer,
               tb_writer: SummaryWriter):
    """Train model for ner_fit_epochs epochs then do self training for self_training_epochs epochs"""

    train_dataset = load_dataset(dataset, dataset_type, tokenizer, args.model_name, args.max_seq_length)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)

    if args.steps_per_epoch == -1 and args.gradient_accumulation_steps == -1:
        raise ValueError('Cannot deduce number of steps per epoch! Set gradient_accumulation_steps or steps_per_epoch!')
    elif args.gradient_accumulation_steps < 0:
        gradient_accumulation_steps = len(train_dataloader) // args.steps_per_epoch
    else:
        gradient_accumulation_steps = args.gradient_accumulation_steps

    min_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    max_steps_per_epoch = min_steps_per_epoch + 1

    st_epochs = args.self_training_epochs
    st_steps = st_epochs * max_steps_per_epoch

    if args.ner_fit_steps < 0:
        ner_epochs = args.ner_fit_epochs
        ner_steps = ner_epochs * max_steps_per_epoch
    else:
        ner_epochs = int(math.ceil(args.ner_fit_steps / min_steps_per_epoch))
        ner_steps = args.ner_fit_steps

    total_steps = ner_steps
    warmup_steps = int(args.warmup_proportion * max_steps_per_epoch)
    warmup_batches = warmup_steps * gradient_accumulation_steps

    # prepare scheduler for NER fitting stage
    model, optimizer, scheduler = initialize_roberta(args, model, total_steps, warmup_steps=warmup_steps,
                                                     end_lr_proportion=args.self_training_lr_proportion)

    # Train!

    global_step = 0
    global_batch = 0

    tr_loss, logging_loss = 0.0, 0.0

    def log_metrics(res: Scores, prefix: str) -> None:
        for metric_name, metric_value in res.items():
            tb_writer.add_scalar(f"{metric_name}_{prefix}", metric_value, global_step)
            tb_writer.add_scalar(f"{metric_name}", metric_value, global_step)

        for group_idx, group in enumerate(optimizer.param_groups):
            tb_writer.add_scalar(f"lr_{group.get('name', f'group{group_idx}')}", group['lr'], global_step)

    # Fitting BERT to NER task

    # warmup epoch
    epoch_iterator = tqdm(train_dataloader, desc=f'Warmup epoch!', total=warmup_batches)
    for batch_idx, batch in enumerate(epoch_iterator):
        if batch_idx >= warmup_batches:
            epoch_iterator.close()
            continue

        batch = tuple(t.to(args.device) for t in batch)
        token_ids, token_mask, attention_mask, labels, label_mask = batch
        inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask,
                  "labels": labels, 'label_mask': label_mask}

        model.train()
        outputs = model(**inputs, self_training=False)
        loss, logits = outputs[0], outputs[1]  # model outputs are always tuple in pytorch-transformers
        loss = loss / gradient_accumulation_steps

        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

    for ner_fit_epoch in range(ner_epochs):

        if args.ner_fit_steps > 0:
            steps_left = args.ner_fit_steps - global_step
            batches_in_current_step = global_batch % gradient_accumulation_steps
            total_batches = min(len(train_dataloader), steps_left * gradient_accumulation_steps - batches_in_current_step)
        else:
            total_batches = len(train_dataloader)

        epoch_iterator = tqdm(train_dataloader, desc=f'Fitting NER on {ner_fit_epoch + 1}/{ner_epochs} epoch', total=total_batches)
        for batch_idx, batch in enumerate(epoch_iterator):

            if batch_idx >= total_batches or 0 < args.ner_fit_steps <= global_step:
                epoch_iterator.close()
                continue

            global_batch += 1

            batch = tuple(t.to(args.device) for t in batch)
            token_ids, token_mask, attention_mask, labels, label_mask = batch
            inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask,
                      "labels": labels, 'label_mask': label_mask}

            model.train()
            outputs = model(**inputs, self_training=False)
            loss, logits = outputs[0], outputs[1]  # model outputs are always tuple in pytorch-transformers
            loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if global_batch % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    # Log metrics
                    results = evaluate(args, model, dataset, DatasetType.VALID, tokenizer)
                    results = {k + '_dev': v for k, v in results.items()}
                    log_metrics({**results, 'loss': (tr_loss - logging_loss) / args.logging_steps}, 'ner')
                    logging_loss = tr_loss

    self_training_teacher_model = deepcopy(model)
    self_training_teacher_model.eval()

    total_steps = st_steps
    args.bert_learning_rate *= args.self_training_lr_proportion
    args.head_learning_rate *= args.self_training_lr_proportion

    # prepare scheduler for NER fitting stage
    model, optimizer, scheduler = initialize_roberta(args, model, total_steps, warmup_steps=0, end_lr_proportion=0)

    # Self training
    for self_training_epoch in range(st_epochs):
        if args.period < 0:
            self_training_teacher_model = deepcopy(model)
            self_training_teacher_model.eval()

        total_batches = len(train_dataloader)
        epoch_iterator = tqdm(train_dataloader, desc=f'Self training on {self_training_epoch + 1}/{st_epochs} epoch', total=total_batches)

        for batch_idx, batch in enumerate(epoch_iterator):

            if batch_idx >= total_batches:
                epoch_iterator.close()
                continue

            global_batch += 1

            if args.period > 0 and global_step % args.period == 0:
                self_training_teacher_model = deepcopy(model)
                self_training_teacher_model.eval()

            batch = tuple(t.to(args.device) for t in batch)

            # Using current teacher to update the label
            token_ids, token_mask, attention_mask, labels, label_mask = batch
            inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask}
            with torch.no_grad():
                outputs = self_training_teacher_model(**inputs)

            predictions = outputs[0]
            if args.correct_frequency:
                pred_labels = soft_frequency(logits=predictions, power=2, probs=self_training_teacher_model.returns_probs)
            else:
                if self_training_teacher_model.returns_probs:
                    pred_labels = predictions
                else:
                    pred_labels = softmax(predictions, dim=-1)

            _threshold = args.label_keep_threshold  # TODO: keep entities with respect to entropy?
            teacher_mask = (pred_labels.max(dim=-1)[0] > _threshold)

            inputs = {**inputs, **{"labels": pred_labels, "label_mask": label_mask & teacher_mask}}

            model.train()
            outputs = model(**inputs, self_training=True, use_kldiv_loss=args.use_kldiv_loss)
            loss, logits = outputs[0], outputs[1]  # model outputs are always tuple in pytorch-transformers

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if global_batch % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    # Log metrics
                    results = evaluate(args, model, dataset, DatasetType.VALID, tokenizer)
                    results = {k + '_dev': v for k, v in results.items()}
                    log_metrics({**results, 'loss': (tr_loss - logging_loss) / args.logging_steps}, 'self_training')
                    logging_loss = tr_loss

        # update lr of all layers
        for group in optimizer.param_groups:
            group['lr'] *= args.lr_st_decay

    tb_writer.close()

    return model, global_step, tr_loss / global_step


def train_nll(args, model: PreTrainedModel, dataset: DatasetName, dataset_type: DatasetType, tokenizer: PreTrainedTokenizer,
              tb_writer: SummaryWriter):
    """Train model for ner_fit_epochs epochs then do self training for self_training_epochs epochs"""
    # TODO
    pass


def train_supervised(args, model: PreTrainedModel, dataset: DatasetName, dataset_type: DatasetType, tokenizer: PreTrainedTokenizer,
                     tb_writer: SummaryWriter):
    """Train model for ner_fit_epochs epochs"""

    train_dataset = load_dataset(dataset, dataset_type, tokenizer, args.model_name, args.max_seq_length)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)

    if args.steps_per_epoch == -1 and args.gradient_accumulation_steps == -1:
        raise ValueError('Cannot deduce number of steps per epoch! Set gradient_accumulation_steps or steps_per_epoch!')
    elif args.gradient_accumulation_steps < 0:
        gradient_accumulation_steps = len(train_dataloader) // args.steps_per_epoch
    else:
        gradient_accumulation_steps = args.gradient_accumulation_steps

    min_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    max_steps_per_epoch = min_steps_per_epoch + 1

    if args.ner_fit_steps < 0:
        ner_epochs = args.ner_fit_epochs
        ner_steps = ner_epochs * max_steps_per_epoch
    else:
        ner_epochs = int(math.ceil(args.ner_fit_steps / min_steps_per_epoch))
        ner_steps = args.ner_fit_steps

    total_steps = ner_steps
    warmup_steps = int(args.warmup_proportion * max_steps_per_epoch)
    warmup_batches = warmup_steps * gradient_accumulation_steps

    model, optimizer, scheduler = initialize_roberta(args, model, total_steps, warmup_steps=warmup_steps)

    # Train!

    global_step = 0
    global_batch = 0

    tr_loss, logging_loss = 0.0, 0.0

    def log_metrics(res: Scores, prefix: str) -> None:
        for metric_name, metric_value in res.items():
            tb_writer.add_scalar(f"{metric_name}_{prefix}", metric_value, global_step)
            tb_writer.add_scalar(f"{metric_name}", metric_value, global_step)

        for group_idx, group in enumerate(optimizer.param_groups):
            tb_writer.add_scalar(f"lr_{group.get('name', f'group{group_idx}')}", group['lr'], global_step)

    # Fitting BERT to NER task

    # warmup epoch
    epoch_iterator = tqdm(train_dataloader, desc=f'Warmup epoch!', total=warmup_batches)
    for batch_idx, batch in enumerate(epoch_iterator):
        if batch_idx >= warmup_batches:
            epoch_iterator.close()
            continue

        batch = tuple(t.to(args.device) for t in batch)
        token_ids, token_mask, attention_mask, labels, label_mask = batch
        inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask,
                  "labels": labels, 'label_mask': label_mask}

        model.train()
        outputs = model(**inputs, self_training=False)
        loss, logits = outputs[0], outputs[1]  # model outputs are always tuple in pytorch-transformers
        loss = loss / gradient_accumulation_steps

        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

    for ner_fit_epoch in range(ner_epochs):

        if args.ner_fit_steps > 0:
            steps_left = args.ner_fit_steps - global_step
            batches_in_current_step = global_batch % gradient_accumulation_steps
            total_batches = min(len(train_dataloader), steps_left * gradient_accumulation_steps - batches_in_current_step)
        else:
            total_batches = len(train_dataloader)

        epoch_iterator = tqdm(train_dataloader, desc=f'Fitting NER on {ner_fit_epoch + 1}/{ner_epochs} epoch', total=total_batches)
        for batch_idx, batch in enumerate(epoch_iterator):

            if batch_idx >= total_batches or 0 < args.ner_fit_steps <= global_step:
                epoch_iterator.close()
                continue

            global_batch += 1

            batch = tuple(t.to(args.device) for t in batch)
            token_ids, token_mask, attention_mask, labels, label_mask = batch
            inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask,
                      "labels": labels, 'label_mask': label_mask}

            model.train()
            outputs = model(**inputs, self_training=False)
            loss, logits = outputs[0], outputs[1]  # model outputs are always tuple in pytorch-transformers
            loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if global_batch % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    # Log metrics
                    results = evaluate(args, model, dataset, DatasetType.VALID, tokenizer)
                    results = {k + '_dev': v for k, v in results.items()}
                    log_metrics({**results, 'loss': (tr_loss - logging_loss) / args.logging_steps}, 'ner')
                    logging_loss = tr_loss

    return model, global_step, tr_loss / global_step


def train(args, model: PreTrainedModel, dataset: DatasetName, dataset_type: DatasetType, training_framework: TrainingFramework,
          tokenizer: PreTrainedTokenizer, tb_writer: SummaryWriter):
    """Train model for ner_fit_epochs epochs then do self training for self_training_epochs epochs"""
    if training_framework == TrainingFramework.BOND:
        return train_bond(args, model, dataset, dataset_type, tokenizer, tb_writer)
    elif training_framework == TrainingFramework.NLL:
        return train_nll(args, model, dataset, dataset_type, tokenizer, tb_writer)
    elif training_framework == TrainingFramework.SUPERVISED:
        args.self_training_epochs = 0
        return train_supervised(args, model, dataset, dataset_type, tokenizer, tb_writer)
    else:
        raise ValueError(f'Unsupported training framework {training_framework.value}!')
