#!/bin/bash
set -e # x
echo "Starting the script..."
# Define variables
NUM_OTUS=$1        # Number of OTUs
NUM_SAMPLES=$2     # Number of individuals (sample size)
NORMALIZATION_CONSTANT=1000
NUMBER_OF_RESPONSES=$3
CODA_METHOD=$4          # Compositional data log-ratio transformation method
OCU_SAMPLING=$7
REALIZATIONS=$5         # Number of experiment realizations per noise level
RESPONSE_BASED=$6       # Response-based flag
SLOPE_VALUES=($8)       # Array of slopes (passed as a string and converted to array)
INTERCEPT_VALUES=($9)   # Array of intercepts (passed as a string and converted to array)
NUM_OCU_ARRAYS=${10}
DEN_OCU_ARRAYS=${11}
MICROBIOME_CASE=${12}      # Microbiome file name
MICROBIOME_FILE_PATH=${13} # Path to microbiome file
DIR_TO_RESULTS=${14}    # Directory to save results
RESPONSE_RMSE_VALUES=(${15})
RMSE_VALUES_OCU_NUMBER=(${16})
RESPONSE_INITIAL_RMSE_VALUES=(${17})
RESPONSE_TAG=${18}
# Set the Internal Field Separator to "-"
IFS='-' read -ra parts <<< "$MICROBIOME_CASE"

# Cast the num_ocu_arrays and den_ocu_arrays values into arrays
IFS=';' read -ra outer_num_ocu_s_list <<< "$NUM_OCU_ARRAYS"
IFS=';' read -ra outer_den_ocu_s_list <<< "$DEN_OCU_ARRAYS"

# Assign the split parts to variables g1, g2, and g3
g1="${parts[0]}"
g2="${parts[1]}"
g3="${parts[2]}"
# Check if the results directory exists; if not, create it
if [ ! -d "$DIR_TO_RESULTS" ]; then
  mkdir -p "$DIR_TO_RESULTS"
  echo "$DIR_TO_RESULTS directory created."
else
  echo "$DIR_TO_RESULTS directory already exists."
fi

# Function to process each combination of sample size, noise level, and realization
process_combination() {
  local noise_level=$1
  local slope=$2
  local intercept=$3
  local num_ocu_s=$4
  local den_ocu_s=$5
  local r=$6
  local noise_level_ocu=$7

  FOLDER_TO_SAVE="$DIR_TO_RESULTS/${noise_level_ocu}otus_${NUM_SAMPLES}samples_${noise_level}noise_${NUMBER_OF_RESPONSES}iterations_${CODA_METHOD}_LR_method_${RESPONSE_BASED}_based_realization${r}"
  CONFIG_FILE_PATH="${FOLDER_TO_SAVE}/config.yaml"

  chmod a+x synthetic_data/analyze_binary_classification/create_synthetic_data_based_real_and_analyze.sh
  noise_level_float=$(echo "$noise_level" | bc -l)
  synthetic_data/analyze_binary_classification/create_synthetic_data_based_real_and_analyze.sh "${NUM_OTUS}"\
  "$NORMALIZATION_CONSTANT" "$FOLDER_TO_SAVE" "${CONFIG_FILE_PATH}" "$NUMBER_OF_RESPONSES"\
  "$noise_level_float" "$slope" "$intercept" "$num_ocu_s" "$den_ocu_s" "$noise_level_ocu"\
  "$CODA_METHOD" "${OCU_SAMPLING}" "$RESPONSE_BASED" "$MICROBIOME_CASE"\
  "$MICROBIOME_FILE_PATH"
}

export -f process_combination

# Loop over all combinations of noise levels and realizations, and run them in parallel

# Maximum number of concurrently processed artificial experiment realizations
MAX_PROCESSES=4
# Function to manage background jobs
wait_for_jobs() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PROCESSES" ]; do
    wait -n  # Wait for any job to finish
  done
}
export -f wait_for_jobs

for i in "${!RESPONSE_RMSE_VALUES[@]}"; do
  noise_level="${RESPONSE_RMSE_VALUES[$i]}"
  slope_value="${SLOPE_VALUES[$i]}"
  intercept_value="${INTERCEPT_VALUES[$i]}"
  num_ocu="${outer_num_ocu_s_list[$i]}"
  den_ocu="${outer_den_ocu_s_list[$i]}"
  noise_level_ocu="${RMSE_VALUES_OCU_NUMBER[$i]}"
  for r in $(seq 1 "$REALIZATIONS"); do
    # Ensure batch processing
    wait_for_jobs
    # Print the current iteration
    echo "Processing noise level: $noise_level, realization: $r"
    process_combination "$noise_level" "$slope_value" "$intercept_value" "$num_ocu" "$den_ocu" "$r" "$noise_level_ocu"&
  done
done
echo "All combinations of noise levels and realizations processed."
# Wait for all background processes to finish
wait

# Direct execution of the Python script to process the results of the artificial data experiments
python3 synthetic_data/analyze_binary_classification/process_auroc_results.py --base_dir "$DIR_TO_RESULTS" \
  --noise_level_ocu_numbers "${RMSE_VALUES_OCU_NUMBER[@]}" \
  --realizations "$REALIZATIONS" \
  --num_samples "$NUM_SAMPLES" \
  --coda_method "$CODA_METHOD" \
  --response_based "$RESPONSE_BASED" \
  --number_of_responses "$NUMBER_OF_RESPONSES" \
  --exp_name "${g1}-${g2}-${g3}"\
  --response_tag "$RESPONSE_TAG" \
  --plot_response_noise "True" \
  --response_rmse_list "${RESPONSE_RMSE_VALUES[@]}" \
  --response_initial_rmse_list "${RESPONSE_INITIAL_RMSE_VALUES[@]}"
