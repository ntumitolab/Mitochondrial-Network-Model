import run

if __name__ == "__main__":

    gaargs = {
        "max_num_iteration": 5,
        "population_size": 5,
        "mutation_probability": 0.6,
        "elit_ratio": 0.03,
        "crossover_probability": 0.7,
        "parents_portion": 0.2,
        "crossover_type": "uniform",
        "max_iteration_without_improv": 5,
    }

    run.run("data/0X_fitting.csv", ga_params=gaargs)
