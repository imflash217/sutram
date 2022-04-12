# Runner

## `EpochBasedRunner`

**Core Logic**

```python
## EpochBasedRunner logic
##
## the execution workflow
## i.e. run 2 epochs for training & 1 epoch for validation ITERATIVELY
workflow = [("train", 2), ("val", 1)]

## condition to stopping
while curr_epoch < max_epochs:
    ## traverse the workflow
    for i, flow in enumerate(workflow):
        ## mode (e.g. "train") determines which function to run
        mode, epochs = flow
        ## epoch_runner will either be self.train() or self.val()
        epoch_runner = getattr(self, mode)
        ## execute the corresponding function
        for _ in range(epochs):
            epoch_runner(data_loaders[i], **kwargs)
...

###################################################################################

def train(self, data_loader, **kwargs):
    ## traverse the data_loader and get batched data for 1 epoch
    for i, data_batch in enumerate(data_loader):
        self.call_hook("before_train_iter")
        self.run_iter(data_batch, train_mode=True, **kwargs)
        self.call_hook("after_train_iter")
    self.call_hook("after_train_epoch")


def val(self, data_loader, **kwargs):
    ## traverse the dataa loader and get a batched data for 1 epoch
    for i, data_batch in enumerate(data_loader):
        self.call_hook("before_val_iter")
        self.run_iter(data_batch, train_mode=False, **kwargs)
        self.call_hook("afetr_val_iter")
    self.call_hook("after_val_epoch")

###################################################################################

```

## `IterBasedRunner`
Different from `EpochBasedRunner`, the workflow in `IterBasedRunner` should be based on iterations.
For example, `[("train", 2), ("val", 1)]` means run 2 iterations for training and 1 iteration for validation iteratively.

**Core Logic**

```python
## Core logic of IterBasedRunner
## Although here we set workflow in iters, we might need the information
## about epochs, which we can provide by IterLoader

## 2 iters for train & 1 iter for validation; iteratively
workflow = [("train", 2), ("val", 1)]

iter_loaders = [IterLoader(x) for x in data_loaders]

## the condition to stop training
while curr_iter < max_iters:
    ## traverse the workflow
    for i, flow in enumerate(workflow):
        mode, iters = flow
        iter_runner = getattr(self, mode)   ## can be self.train() or self.val()
        for _ in range(iters):
            iter_runner(iter_loaders[i], **kwargs)

...

###################################################################################

def train(self, data_loader, **kwargs):
    data_batch = next(data_loader)
    self.call_hook("before_train_iter")
    self.outputs = self.model.train_step(data_batch, train_mode=True, **kwargs)
    self.call_hook("after_train_iter")

def val(self, data_loader, **kwargs):
    data_batch = next(data_loader)
    self.call_hook("before_val_iter")
    self.outputs = self.model.val_step(data_batch, train_mode=False, **kwargs)
    self.call_hook("after_val_iter")

###################################################################################

```

Other than the above moethods, `EpochBasedRunner` and `IterBasedRunner` provide useful methods like
`resume()`, `save_checkpoin()` and `register_hook()`


## A simple example

Here, we walk-through the usage of runner for a classification task.

### Step-1. Initialize the `dataloaders`, `model`, `optimizers` etc...

```python
## 🎯 Step-1
## initialize model
model = ...

## initialize optimizer
## typically we set cfg.optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=1e-4)
optimizer = build_optimizer(model, cfg.optimizer)

## create dataloaders
data_loaders = [
    build_dataloader(
        ds, 
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        ...
    ) for ds in dataset
]

```

### Step-2: Initialize Runner

```python
## 🎯 Step-2

runner = build_runner(
    ## cfg.runner is typically set as
    ## runner = dict(type="EpochBasedRunner", max_epochs=200)
    cfg.runner,
    default_args=dict(
        model=model,
        batch_processor=None,
        optimizer=optimizer,
        logger=logger,
    ),
)

```

### Step-3: Register Training hooks and customized hooks

```python
## 🎯 Step-3
## Register default hooks necessary for training
runner.register_trainig_hooks(
    cfg.lr_config,          ## e.g. lr_config=dict(policy="step", step=[100. 150])
    optimizer_config,       ## e.g. grad_clip
    cfg.checkpoint_config,  ## e.g. checkpoint_config=dict(interval=1) -> i.e. saving every epoch
    cfg.log_config,
)

## Register custom hooks
## e.g. if we want to enable EMA (exp. moving avg.) then we should set 
## custom_hooks=[dict(type="EMAHook")]

if cfg.get("custom_hooks", None):
    custom_hooks = cfg.custom_hooks
    for hook_cfg in custom_hooks:
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop("priority", "NORMAL")
        hook = build_from_cfg(hook_cfg, HOOKS)
        runner.register_hook(hook, priority=priority)

```

Then, we can use `resume` and `load_checkpoint` to load existing weights.

### Step-4: Start training

```python
## workflow is typically set inside the config as below
## workflow = [("train", 2), ("val", 1)]

## 🎯 Start the runner for training
runner.run(data_loaders, cfg.workflow)

```

