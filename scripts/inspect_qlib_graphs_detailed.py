
import sys
import inspect
from qlib.contrib.report import analysis_position as qap
from qlib.contrib.report import analysis_model as qam

def inspect_module(mod, name_prefix):
    for name, obj in inspect.getmembers(mod):
        if inspect.isfunction(obj) and (name.endswith('_graph') or name == 'risk_analysis_graph'):
            print(f"\n{'='*20} {name_prefix}.{name} {'='*20}")
            print(f"Signature: {inspect.signature(obj)}")
            doc = inspect.getdoc(obj)
            if doc:
                print(f"Docstring:\n{doc}")
            else:
                print("Docstring: None")
            
            # Print first 20 lines of source code as a hint to implementation
            try:
                source_lines = inspect.getsourcelines(obj)[0]
                print("\nSource Code Snippet (first 30 lines):")
                print("".join(source_lines[:30]))
            except Exception as e:
                print(f"Could not read source: {e}")

print(">>> Inspecting analysis_position graphs...")
inspect_module(qap, "analysis_position")

print("\n\n>>> Inspecting analysis_model graphs...")
inspect_module(qam, "analysis_model")
