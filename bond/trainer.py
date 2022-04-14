from copy import deepcopy
from enum import Enum
from typing import List

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from bond.data import DatasetName, DatasetType, load_dataset, load_tags_dict, load_transformed_dataset, BertExample
from bond.utils import Scores, initialize_roberta, ner_scores, soft_frequency, convert_hard_to_soft_labels

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class TrainingFramework(Enum):
    BOND = 'bond'
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
        batch: BertExample = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            token_ids, token_mask, attention_mask, labels, label_mask, gold_label_mask, weight = batch
            inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask,
                      "labels": labels, 'label_mask': label_mask, 'seq_weights': weight, "gold_label_mask": gold_label_mask}
            with torch.cuda.amp.autocast():
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
               tb_writer: SummaryWriter, amp_scaler: torch.cuda.amp.GradScaler):
    """Train model for ner_fit_epochs epochs then do self training for self_training_epochs epochs"""
    if args.add_gold_labels > 0.0 and dataset_type == DatasetType.DISTANT:
        train_dataset = load_transformed_dataset(dataset, args.add_gold_labels, tokenizer, args.model_name, args.max_seq_length)
    else:
        train_dataset = load_dataset(dataset, dataset_type, tokenizer, args.model_name, args.max_seq_length)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    num_labels = len(load_tags_dict(dataset).keys())

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
    model, optimizer, scheduler = initialize_roberta(args, model, total_steps, warmup_steps=warmup_steps,
                                                     end_lr_proportion=args.self_training_lr_proportion)

    # Train!

    global_batch = 0
    examples_from_last_log = 0
    steps_from_last_log = 0

    tr_loss, logging_loss = 0.0, 0.0

    def log_metrics(res: Scores, prefix: str) -> None:
        examples_seen = global_batch * args.batch_size

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

        batch: BertExample = tuple(t.to(args.device) for t in batch)
        token_ids, token_mask, attention_mask, labels, label_mask, gold_label_mask, weight = batch
        inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask,
                  "labels": labels, 'label_mask': label_mask, 'seq_weights': weight, "gold_label_mask": gold_label_mask}

        model.train()
        with torch.cuda.amp.autocast():
            outputs = model(**inputs, self_training=False, warmup=True)
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

    for ner_fit_epoch in range(ner_epochs):

        total_batches = len(train_dataloader)

        epoch_iterator = tqdm(train_dataloader, desc=f'Fitting NER on {ner_fit_epoch + 1}/{ner_epochs} epoch', total=total_batches)
        for batch_idx, batch in enumerate(epoch_iterator):
            global_batch += 1
            examples_from_last_log += args.batch_size

            batch: BertExample = tuple(t.to(args.device) for t in batch)
            token_ids, token_mask, attention_mask, labels, label_mask, gold_label_mask, weight = batch
            inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask,
                      "labels": labels, 'label_mask': label_mask, 'seq_weights': weight, "gold_label_mask": gold_label_mask}

            model.train()
            with torch.cuda.amp.autocast():
                outputs = model(**inputs, self_training=False)
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
                results = evaluate(args, model, dataset, DatasetType.VALID, tokenizer)
                results = {k + '_dev': v for k, v in results.items()}
                log_metrics({**results, 'loss': (tr_loss - logging_loss) / steps_from_last_log}, 'ner')
                logging_loss = tr_loss
                examples_from_last_log = 0
                steps_from_last_log = 0

    self_training_teacher_model = deepcopy(model)
    self_training_teacher_model.eval()

    total_steps = st_steps
    args.bert_learning_rate *= args.self_training_lr_proportion
    args.head_learning_rate *= args.self_training_lr_proportion

    # prepare scheduler for self training stage
    model, optimizer, scheduler = initialize_roberta(args, model, total_steps, warmup_steps=0, end_lr_proportion=0)
    model.prepare_for_self_training()

    # Self training
    for self_training_epoch in range(st_epochs):

        total_batches = len(train_dataloader)
        batches_per_update = total_batches // args.updates
        epoch_iterator = tqdm(train_dataloader, desc=f'Self training on {self_training_epoch + 1}/{st_epochs} epoch', total=total_batches)

        for batch_idx, batch in enumerate(epoch_iterator):

            if batch_idx >= total_batches:
                epoch_iterator.close()
                continue

            if batch_idx % batches_per_update == 0:
                self_training_teacher_model = deepcopy(model)
                self_training_teacher_model.eval()

            global_batch += 1
            examples_from_last_log += args.batch_size

            batch: BertExample = tuple(t.to(args.device) for t in batch)

            # Using current teacher to update the labels
            token_ids, token_mask, attention_mask, labels, label_mask, gold_label_mask, weight = batch
            inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask,
                      "gold_label_mask": gold_label_mask}
            with torch.no_grad():
                outputs = self_training_teacher_model(**inputs)

            predictions = outputs[0]
            if args.correct_frequency:
                pred_labels = soft_frequency(logits=predictions, power=2, probs=True)
            else:
                pred_labels = predictions

            _threshold = args.label_keep_threshold
            teacher_mask = (pred_labels.max(dim=-1)[0] > _threshold)

            pred_labels[gold_label_mask] = convert_hard_to_soft_labels(labels[gold_label_mask], num_labels)

            inputs = {**inputs,
                      **{"labels": pred_labels, "label_mask": (label_mask & teacher_mask) | gold_label_mask, 'seq_weights': weight}}

            model.train()
            with torch.cuda.amp.autocast():
                outputs = model(**inputs, self_training=True, use_kldiv_loss=args.use_kldiv_loss)
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
                results = evaluate(args, model, dataset, DatasetType.VALID, tokenizer)
                results = {k + '_dev': v for k, v in results.items()}
                log_metrics({**results, 'loss': (tr_loss - logging_loss) / steps_from_last_log}, 'self_training')
                logging_loss = tr_loss
                examples_from_last_log = 0
                steps_from_last_log = 0

    tb_writer.close()

    return model


def train_supervised(args, model: PreTrainedModel, dataset: DatasetName, dataset_type: DatasetType, tokenizer: PreTrainedTokenizer,
                     tb_writer: SummaryWriter, amp_scaler: torch.cuda.amp.GradScaler):
    """Train model for ner_fit_epochs epochs"""

    if args.add_gold_labels > 0.0 and dataset_type == DatasetType.DISTANT:
        train_dataset = load_transformed_dataset(dataset, args.add_gold_labels, tokenizer, args.model_name, args.max_seq_length)
    else:
        train_dataset = load_dataset(dataset, dataset_type, tokenizer, args.model_name, args.max_seq_length)

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

        batch: BertExample = tuple(t.to(args.device) for t in batch)
        token_ids, token_mask, attention_mask, labels, label_mask, gold_label_mask, weight = batch
        inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask,
                  "labels": labels, 'label_mask': label_mask, 'seq_weights': weight, "gold_label_mask": gold_label_mask}

        model.train()
        with torch.cuda.amp.autocast():
            outputs = model(**inputs, self_training=False, warmup=True)
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

    for ner_fit_epoch in range(ner_epochs):

        total_batches = len(train_dataloader)

        epoch_iterator = tqdm(train_dataloader, desc=f'Fitting NER on {ner_fit_epoch + 1}/{ner_epochs} epoch', total=total_batches)
        for batch_idx, batch in enumerate(epoch_iterator):

            global_batch += 1
            examples_from_last_log += args.batch_size

            batch: BertExample = tuple(t.to(args.device) for t in batch)
            token_ids, token_mask, attention_mask, labels, label_mask, gold_label_mask, weight = batch
            inputs = {"input_ids": token_ids, "token_mask": token_mask, "attention_mask": attention_mask,
                      "labels": labels, 'label_mask': label_mask, 'seq_weights': weight, "gold_label_mask": gold_label_mask}

            model.train()
            with torch.cuda.amp.autocast():
                outputs = model(**inputs, self_training=False)
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
                results = evaluate(args, model, dataset, DatasetType.VALID, tokenizer)
                results = {k + '_dev': v for k, v in results.items()}
                log_metrics({**results, 'loss': (tr_loss - logging_loss) / steps_from_last_log}, 'ner')
                logging_loss = tr_loss
                examples_from_last_log = 0
                steps_from_last_log = 0

    return model


def train(args, model: PreTrainedModel, dataset: DatasetName, dataset_type: DatasetType, training_framework: TrainingFramework,
          tokenizer: PreTrainedTokenizer, tb_writer: SummaryWriter, amp_scaler: torch.cuda.amp.GradScaler):
    """Train model for ner_fit_epochs epochs then do self training for self_training_epochs epochs"""
    if training_framework == TrainingFramework.BOND:
        return train_bond(args, model, dataset, dataset_type, tokenizer, tb_writer, amp_scaler)
    elif training_framework == TrainingFramework.SUPERVISED:
        args.self_training_epochs = 0
        return train_supervised(args, model, dataset, dataset_type, tokenizer, tb_writer, amp_scaler)
    else:
        raise ValueError(f'Unsupported training framework {training_framework.value}!')
