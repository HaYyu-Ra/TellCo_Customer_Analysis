import argparse
from acis.analysis import run_analysis
p=argparse.ArgumentParser()
p.add_argument("--data",default="data/raw/MachineLearningRating_v3.txt")
p.add_argument("--output",default="reports/generated")
a=p.parse_args()
r=run_analysis(a.data,a.output)
print("Policies:",r["descriptive"]["unique_policies"])
print("Tests:",r["tests"])
print("Models:",r["models"])
