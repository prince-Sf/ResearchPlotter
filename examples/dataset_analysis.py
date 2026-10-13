from research_plotter.pipelines.dataset_analysis import analyze_dataset
from research_plotter.pipelines.figure_generation import recommend_dataset_figures

frame, profile, report = analyze_dataset("data/examples/measurements.csv")
print(profile.model_dump_json(indent=2))
print(recommend_dataset_figures(frame, profile))
