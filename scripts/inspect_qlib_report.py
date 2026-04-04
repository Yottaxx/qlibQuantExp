
import qlib.contrib.report.analysis_model as qam
import inspect

print("Function: model_performance_graph")
print(inspect.getsource(qam.model_performance_graph))

print("\n" + "="*50 + "\n")
if hasattr(qam, "cumulative_return_graph"):
    print("Function: cumulative_return_graph")
    print(inspect.getsource(qam.cumulative_return_graph))
