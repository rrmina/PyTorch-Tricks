# Reusable PyTorch codes

## Calculate Number of Parameters
```python
# Total Number of Parameters
total_params = sum(p.numel() for p in model.parameters())

# Total Number of Trainable Parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## ImageFolder with Folder Paths

```python
class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (*original_tuple, path)
        return tuple_with_path
```

## Reshape Layer for nn.Sequential

```python
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
```

## Debugger Layer for nn.Sequential
```python
class Debugger(nn.Module):
    def __init__(self):
        super(Debugger, self).__init__()
    
    def forward(self, x):
        print(x)
        return x
```
