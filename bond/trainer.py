import math
from copy import deepcopy
from typing import List

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from bond.data import DatasetName, DatasetType, load_dataset, load_tags_dict
from bond.model import JunctionStrategy
from bond.utils import Scores, initialize_roberta, ner_scores, soft_frequency

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def evaluate(args, model: PreTrainedModel, dataset: DatasetName, dataset_type: DatasetType, tokenizer: PreTrainedTokenizer) -> Scores:
    eval_dataset = load_dataset(dataset, dataset_type, tokenizer, args.model_name, args.max_seq_length,
                                junction_strategy=JunctionStrategy(args.junction_strategy))
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    predicted_labels: List[int] = []
    true_labels: List[int] = []

    model.eval()
    for batch in tqdm(eval_dataloader, desc=f"Evaluating on {dataset_type.value} dataset", leave=False):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "labels": batch[1], 'label_mask': batch[2], "attention_mask": batch[3]}
            outputs = model(**inputs)

            tmp_eval_loss, logits, label_mask = outputs[:3]
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1

        batch_raveled_logits = torch.cat(list(logits.detach().cpu()), dim=0)
        batch_raveled_output_label_mask = torch.cat(list(label_mask.cpu()), dim=0)
        batch_raveled_input_label_mask = torch.cat(list(inputs['label_mask'].cpu()), dim=0)
        batch_raveled_true_labels = torch.cat(list(inputs['labels'].cpu()), dim=0)
        batch_predicted_labels = torch.argmax(batch_raveled_logits, dim=-1).masked_select(batch_raveled_output_label_mask > 0)
        batch_true_labels = batch_raveled_true_labels.masked_select(batch_raveled_input_label_mask > 0)

        assert len(batch_predicted_labels) == len(batch_true_labels)

        predicted_labels.extend(batch_predicted_labels.tolist())
        true_labels.extend(batch_true_labels.tolist())

    eval_loss = eval_loss / nb_eval_steps
    results = ner_scores(true_labels, predicted_labels, load_tags_dict(dataset))
    results['loss'] = eval_loss

    return results


def train(args, model: PreTrainedModel, dataset: DatasetName, tokenizer: PreTrainedTokenizer, tb_writer: SummaryWriter):
    """Train model for ner_fit_epochs epochs then do self training for self_training_epochs epochs"""

    train_dataset = load_dataset(dataset, DatasetType.DISTANT, tokenizer, args.model_name, args.max_seq_length,
                                 junction_strategy=JunctionStrategy(args.junction_strategy))

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    if args.steps_per_epoch == -1 and args.gradient_accumulation_steps == -1:
        raise ValueError('Cannot deduce number of steps per epoch! Set gradient_accumulation_steps or steps_per_epoch!')
    elif args.gradient_accumulation_steps < 0:
        gradient_accumulation_steps = len(train_dataloader) // args.steps_per_epoch
    else:
        gradient_accumulation_steps = args.gradient_accumulation_steps

    max_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps

    st_epochs = args.self_training_epochs
    ner_epochs = args.ner_fit_epochs if args.ner_fit_steps < 0 else int(math.ceil(args.ner_fit_steps / max_steps_per_epoch))

    model, optimizer, scheduler = initialize_roberta(args, model)

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
            inputs = {"input_ids": batch[0], "labels": batch[1], "label_mask": batch[2], "attention_mask": batch[3]}

            model.train()
            outputs = model(**inputs, self_training=False)
            loss, logits, final_embeds = outputs[0], outputs[1], outputs[2]  # model outputs are always tuple in pytorch-transformers
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
            inputs = {"input_ids": batch[0], "label_mask": batch[2], "attention_mask": batch[3]}
            with torch.no_grad():
                outputs = self_training_teacher_model(**inputs)

            pred_labels = outputs[0]
            if args.correct_frequency:
                pred_labels = soft_frequency(logits=pred_labels, power=2, probs=self_training_teacher_model.returns_probs)

            _threshold = args.label_keep_threshold  # TODO: keep entities with respect to entropy?
            teacher_mask = (pred_labels.max(dim=-1)[0] > _threshold)

            inputs = {"input_ids": batch[0], "labels": pred_labels, "label_mask": batch[2] | teacher_mask,  "attention_mask": batch[3]}

            model.train()
            outputs = model(**inputs, self_training=True, use_kldiv_loss=args.use_kldiv_loss)
            loss, logits, final_embeds = outputs[0], outputs[1], outputs[2]  # model outputs are always tuple in pytorch-transformers

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if global_batch % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
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
