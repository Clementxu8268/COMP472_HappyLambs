import os

def save_results_as_text(results, model_name, directory="results"):
    """
    Save results as text file
    """
    # Create dir if it is not there
    os.makedirs(directory, exist_ok=True)

    # Define save path
    results_path = os.path.join(directory, f"{model_name}_results.txt")

    # Write down the result
    with open(results_path, 'w') as f:
        f.write(f"Results for {model_name}:\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print(f"Results saved as {results_path}")
