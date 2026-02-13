## How the generate_synthetic_data module works

- Generates microbiome and response data according to the parameter values in the header of the module;
- Stores all the files in the folder called `simulated`;
- The folder includes 4 microbiome-related files: composition, counts, correlation matrix, covariance matrix;
- The folder includes 4 X total number of noise levels response-related files: two response files, CLR-like and pairs-like, and corresponding info files.

## How you can use the generate_synthetic_data module

- Optional: adjust the parameter values in the header of the module;
- Run it;
- Choose `synthetic_compositional_data_....tsv` as microbiome compositional data and the response files which fits your goal and copy them to your dataset directory;
- Rename the files accordingly to fit your config file;
- Run `CompoRes`.