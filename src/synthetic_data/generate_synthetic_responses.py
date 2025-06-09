import os
import os.path

import numpy as np
import pandas as pd


def generate_microbiome_absolute_count(compositional_microbiome):
    """
    Generate absolute count from relative abundance
    :param compositional_microbiome: relative abundance of microbiome
    :return: absolute count of microbiome df
    """
    absolute_counts = compositional_microbiome * np.random.lognormal(4.6, 0.1,
                                                                     compositional_microbiome.shape)
    pd.DataFrame(absolute_counts).to_csv("absolute_counts.tsv", sep="\t")
    return absolute_counts


def create_multiple_responses(noise_level: float, slope: float, intercept: float,
                              num_ocu_s_str: str, den_ocu_s_str: str,
                              microbiome_file_path: str, folder_to_save: str, name: str = "",
                              correlation="correlated", response_based="balance", num_of_responses=1):
    """
        Generate multiple synthetic responses based on microbiome data.

        This function creates multiple synthetic responses by generating random combinations of taxa,
        computing responses using the specified parameters, and adding Gaussian noise. The responses
        are aligned with the microbiome data index and saved to output files.

        :param noise_level: The effect of the gaussian noise to be added to the responses.
        :param slope: The linear constant to multiply the balance values by.
        :param intercept: The intercept constant to add to the responses.
        :param num_ocu_s_str: The labels of the numerator taxa passed as a list cast into a string.
        :param den_ocu_s_str: The labels of the denominator taxa passed as a list cast into a string.
        :param microbiome_file_path: Path to the microbiome data file.
        :param folder_to_save: Directory path where the generated responses and taxa information will be saved.
        :param name: A base name for naming the generated responses.
        :param correlation: Specifies if the response should be "correlated" or "uncorrelated".
        :param response_based: Determines if the response is based on "balance" or "taxon".
        :param num_of_responses: The number of synthetic responses to generate.
        :return: None. The function saves the generated responses and taxa information to TSV files.
        """
    # Load the microbiome file to get its index
    microbiome_df = pd.read_csv(microbiome_file_path, sep="\t", index_col=0)
    microbiome_index = microbiome_df.index
    taxa_info = pd.DataFrame(columns=["Response_Name", "Num", "Den"])
    all_responses = pd.DataFrame()
    for i in range(num_of_responses):

        num = num_ocu_s_str[1:-1].split(",")
        den = den_ocu_s_str[1:-1].split(",")

        res_name = f"{name}_{i}"
        response = create_response_based_data(num, den, noise_level, slope, intercept,
                                              microbiome_file_path, folder_to_save,
                                              correlation, response_based)
        # Convert response Series to DataFrame and rename column
        response = response.to_frame(name=res_name)
        # Set the index of the response to match the microbiome file's index
        response.index = microbiome_index
        all_responses = pd.concat([all_responses, response], axis=1)
        # Store the num and den values with the response name
        new_row = {"Response_Name": res_name, "Num": num, "Den": den}
        # Append the new row to the DataFrame using pd.concat
        taxa_info = pd.concat([taxa_info, pd.DataFrame([new_row])], ignore_index=True)

    all_responses.to_csv(f"{folder_to_save}/{name}.tsv", sep="\t", index=True)
    # Save taxa info to a TSV file
    taxa_info.to_csv(f"{folder_to_save}/{name}_taxa_info.tsv", sep="\t", index=True)


def create_response_based_data(num, den, noise_level: float, linear_const: float, intercept_const: float,
                               microbiome_file_path: str, folder_to_save: str,
                               correlation="correlated", response_based="balance") -> pd.Series:
    """
    Generate a response based on the specified parameters.

    This function generates a synthetic response by adding Gaussian noise to computed values
    derived from microbiome data, depending on the specified correlation type and response basis.


    :param num: Index of the numerator taxon.
    :param den: Index of the denominator taxon.
    :param noise_level: The standard deviation of the Gaussian noise to be added.
    :param linear_const: The linear constant to multiply the balance values by.
    :param intercept_const: The intercept constant to add to the response.
    :param microbiome_file_path: Path to the microbiome data file.
    :param folder_to_save: Directory path where the results will be saved.
    :param correlation: Specifies if the response should be "correlated" or "uncorrelated".
    :param response_based: Determines if the response is based on "balance" or "taxon".
    :return: A Pandas Series containing the generated response values.
    """
    samples = pd.read_csv(microbiome_file_path, sep='\t', index_col=0)

    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    response = None
    noise = noise_level * np.random.default_rng().standard_normal(samples.shape[0])  # Gaussian noise
    if response_based == "balance":
        if correlation == "correlated":
            balance = np.sqrt(0.5) * np.log(samples[num].sum(axis=1) / samples[den].sum(axis=1))
            response = intercept_const + np.abs(balance) * linear_const + noise
        elif correlation == "uncorrelated":
            response = pd.Series(noise)

    elif response_based == "taxon":
        if correlation == "correlated":
            absolute_microbiome_data = generate_microbiome_absolute_count(samples)
            response = absolute_microbiome_data[num].sum(axis=1) * linear_const + intercept_const + noise
        elif correlation == "uncorrelated":
            response = pd.Series(noise)
    return response
