# PyProNet

## Overview

**PyProNet** is a Python script designed for network analysis and manipulation using libraries such as pandas, networkx, and numpy. This script includes functionality for setting up and customizing logging to help with debugging and monitoring the execution of the script.

## Features

- Initialization and configuration of logging for both console and file outputs.
- Customizable logging levels and formats.
- Utilities for network analysis using networkx.

## Requirements

- Python 3.12.1 or higher
- `pandas` 2.2.2 or higher
- `networkx` 3.3 or higher
- `numpy` 2.0 or higher

Install the required packages with:

```bash
pip install pandas networkx numpy
```

## Usage

### Logging Configuration

The script sets up a logger with both console and file handlers. By default, logs are written to `file.log` in the current working directory. The logging levels and formats can be customized using the `change_logger` function.

#### Example

```python
import pypronet as ppn

ppn.change_logger(log_file_path: str = "../file.log", 
                 console_level: int = logging.WARNING,
                 console_format: str = "Module: %(name)s, function: %(funcName)s - %(message)s",
                 file_level: int = logging.DEBUG,
                 file_format: str = "%(asctime)s - %(levelname)s - Module: %(name)s, function: %(funcName)s - %(message)s"
                 )
```

### Network Analysis

The script utilizes networkx for network operations. Ensure you have your data prepared and loaded correctly to leverage these functionalities.

#### Example

```python
import pypronet as ppn

# Create a scalefree network with 10 nodes
G = ppn.create_scalefree(maxnodes=10, step_edges=2)

# Print the density of the network
print(ppn.density(graph=G))
```

### License

This project is licensed under the MIT License. See `LICENSE` for more information.
