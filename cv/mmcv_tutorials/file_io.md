# File IO
`MMCV` provides universal API that supports:
- loading data from multiple backends.
- dumping data to multiple backends.

The file type can be any of the following:
- `.json`
- `.yaml`
- `.pkl`

## Load from disk **or** dump to disk

```python
import mmcv

#########################################################
## load from disk
##
## Case-1: load data from a file
data = mmcv.load("test.json")
data = mmcv.load("test.yaml")
data = mmcv.load("test.pkl")

## Case-2: load from a file-like object
with open("test.json", "r") as f:
    data = mmcv.load(f, file_format="json")

#########################################################
## dump to disk
##
## Case-1: dump data to a string
json_str = mmcv.dump(data, file_format="json")

## Case-2: dump data to a file on disk (infer format from the file-extension)
mmcv.dump(data, "out.pkl")  ## here the data is written as a pickle file

## Case-3: dump data to a file on disk using a file-like object
with open("out.yaml", "w") as f:
    data = mmcv.dump(data, f, file_format="yaml")

```

















