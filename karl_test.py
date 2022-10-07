import composer, torch, transformers

class train_state:
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        actiation_function = 'relu',
        bos_token_id = ord(';'),
        eos_token_id = ord(';'),
        n_embd = 12,
        n_head = 3,
        n_layer = 1,
        n_positions = 1024,
        vocab_size = 256
    ))
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3),
    ]
    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(
            optimizer = optimizers[0],
            gamma = 0.9,
            verbose = True
        ),
    ]
    batch = None

# for alibi:
# sequence lengths should be > 256
# train_sequence_length_scaling should be > 0.03125
# sequence lengths may be scaled by that sequence length scaling.
#hrr = composer.algorithms.HRRAlibi(max_sequence_length=8192, train_sequence_length_scaling=0.0625)
hrr = composer.algorithms.HRRAlibi(max_sequence_length=1024, train_sequence_length_scaling=0.25)

hrr.apply(composer.core.Event.INIT, train_state, None)

# ==> state.model should now be a transformer model that uses both algorithms.

### thinking of data generation.
### we can do summation of random binary values
### since the code is cpu, no point to generating on-gpu
### but we might want to cache them
def make_batch(length = hrr.max_sequence_length, size = 1):
    left, right = torch.randint(-1<<16,1<<16,(2,))
    batch = dict(
        input_ids = [],
        labels = []
    )
    for row_idx in range(size):
        input_ids = []
        labels = []
        length_so_far = 0
        while length_so_far < length:
            inputs = torch.tensor(tuple(f'{left:b}+{right:b}='.encode()))
            outputs = torch.tensor(tuple(f'{left+right:b};'.encode()))
            input_ids.extend((inputs, outputs))
            labels.extend((torch.full((len(inputs),), -100), outputs))
            length_so_far += len(inputs) + len(outputs)
        batch['input_ids'].append(torch.cat(input_ids)[:length])
        batch['labels'].append(torch.cat(labels)[:length])

    batch['input_ids'] = torch.stack(batch['input_ids'])
    batch['labels'] = torch.stack(batch['labels'])

    return batch

train_state.model.train()
for epoch in range(65):
    total_loss = 0
    num_batches = 16
    for batch in range(num_batches):
        train_state.model.zero_grad()
        train_state.batch = make_batch(size=1)
        hrr.apply(composer.core.Event.AFTER_DATALOADER, train_state, None)
        train_state.outputs = train_state.model(**train_state.batch)
        train_state.outputs.loss.backward()
        for optimizer in train_state.optimizers:
            optimizer.step()
        total_loss += train_state.outputs.loss.detach()
    print('loss', total_loss / num_batches)
    for scheduler in train_state.schedulers:
        scheduler.step()
