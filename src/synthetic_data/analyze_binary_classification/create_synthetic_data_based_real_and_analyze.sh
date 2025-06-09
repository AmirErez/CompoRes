#!/bin/bash
# set -ex
# Define variables
FOLDER_TO_SAVE=$3
CONFIG_FILE_PATH=$4
NUMBER_OF_RESPONSES=$5
noise_level_float=$6
slope=$7
intercept=$8
num_ocu_s=$9
den_ocu_s=${10}
noise_level_ocu_number=${11}
CODA_METHOD=${12}
OCU_SAMPLING_RATE=${13}
RESPONSE_BASED=${14}
MICROBIOME_NAME=${15}
MICROBIOME_FILE_PATH=${16}

# Create results folder if it doesn't exist
mkdir -p "${FOLDER_TO_SAVE}"
mkdir -p "${FOLDER_TO_SAVE}/microbiome"
mkdir -p "${FOLDER_TO_SAVE}/response"

# copy file if it doesn't exist
cp -n "$MICROBIOME_FILE_PATH" "$FOLDER_TO_SAVE"/"$MICROBIOME_NAME".tsv

# Set the Internal Field Separator to "-"
IFS='-' read -ra parts <<< "$MICROBIOME_NAME"

# Assign the split parts to variables g1, g2, and g3
g1="${parts[0]}"
g2="${parts[1]}"
g3="${parts[2]}"


# Function to update the config file
update_config_file() {
    local config_path=$1
#    local microbiome_file_path=$2
#    local response_file_path=$3
    local group1=$2
    local group2=$3
    local group3=$4
    local coda_method=$5

    cat <<EOT > $config_path
# Please check the parameters, and adjust them according to your circumstances

# Project name
PROJECT: test
# ================== Control of the workflow ==================
# Enter the project experiment parameters
GROUP1: $group1

GROUP2: $group2

GROUP3: $group3

# Paths to files
PATH_TO_MICROBIOME: $FOLDER_TO_SAVE/microbiome
PATH_TO_RESPONSE: $FOLDER_TO_SAVE/response
PATH_TO_OUTPUTS: $FOLDER_TO_SAVE/compores_output
PATH_TO_METADATA: $FOLDER_TO_SAVE/metadata

# OCU sampling rate
OCU_SAMPLING_RATE: $OCU_SAMPLING_RATE

# Log-ratio transformation method applied for compositional data analysis (choose one from:"pairs","CLR")
CODA_METHOD: $coda_method

# Correlation calculation method (choose one from:"spearman","pearson")
CORR: pearson

# Sample shuffling method ("response" or "microbiome"), number of shuffles per cycle, number of shuffling cycles
SHUFFLE: microbiome
N_SHUFFLES: 10
SHUFFLE_CYCLES: 5

# Maximum number of workers for parallel processing
N_WORKERS:
EOT
}

export PYTHONPATH="."

# Loop over treatment groups and iterations
for treatment_group in "correlated" "uncorrelated"; do
    cp "$MICROBIOME_FILE_PATH" "$FOLDER_TO_SAVE"/microbiome/${treatment_group}-$g2-$g3.tsv

    response_name="${treatment_group}-${g2}"
    response_file_path="${FOLDER_TO_SAVE}/response"

    # Call Python script to generate synthetic data
    python3 <<EOF
import sys
import os

# Print the current working directory
print("Current working directory:", os.getcwd())

sys.path.append('synthetic_data')  # Adjust this path as necessary

# Debugging information
print("CALLING GENERATING DATA")

from generate_synthetic_responses import create_multiple_responses

create_multiple_responses(${noise_level_float}, ${slope}, ${intercept}, "${num_ocu_s}", "${den_ocu_s}", "${MICROBIOME_FILE_PATH}",
                               "${response_file_path}", "${response_name}", "${treatment_group}","${RESPONSE_BASED}", ${NUMBER_OF_RESPONSES})
EOF

    # Update config file for this iteration
    update_config_file "${CONFIG_FILE_PATH}" ${treatment_group} "${g2}" "${g3}" "${CODA_METHOD}"

    # Run the CompoRes script
    python3 -m compores --config "${CONFIG_FILE_PATH}" --no_plotting --ocu_case "${noise_level_ocu_number}"

done
# wait

# Python script for plotting
python3 <<EOF
import sys
sys.path.append('synthetic_data/analyze_binary_classification')  # Replace with your actual path
from run_binary_classification_analysis import plot_p_value_frequencies, plot_p_value_boxplot, plot_roc_curve, read_p_values_from_dictionaries

correlated_microbiome_dir_path =  f"${FOLDER_TO_SAVE}/compores_output/compores_basic_results/correlated-${g2}-${g3}/${CODA_METHOD}"
uncorrelated_microbiome_dir_path = f"${FOLDER_TO_SAVE}/compores_output/compores_basic_results/uncorrelated-${g2}-${g3}/${CODA_METHOD}"

c_p_value_dict_path =f"{correlated_microbiome_dir_path}/p_values.pkl"
u_p_value_dict_path =f"{uncorrelated_microbiome_dir_path}/p_values.pkl"

try:
  c_p_values, c_labels, u_p_values, u_labels = read_p_values_from_dictionaries(
  c_p_value_dict_path, u_p_value_dict_path, ${noise_level_ocu_number}
  )
except FileNotFoundError:
  print("'p_value.pkl' file not found either for uncorrelated or for correlated synthetic responses: some intermediate results seem to be lost; re-run the job after investigating the issue and implementing relevant adjustments.")
  sys.exit(1)
plot_p_value_frequencies(c_p_values, "correlated_response", u_p_values, "uncorrelated_response", "${FOLDER_TO_SAVE}")
plot_p_value_boxplot(c_p_values, "correlated_response", u_p_values, "uncorrelated_response", "${FOLDER_TO_SAVE}")
plot_roc_curve(c_p_values, c_labels, u_p_values, u_labels, "${FOLDER_TO_SAVE}", "${g1}-${g2}-${g3}")
EOF
