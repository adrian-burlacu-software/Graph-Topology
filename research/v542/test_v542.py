import importlib.util, sys
spec=importlib.util.spec_from_file_location('v542','graph_knowledge_discovery.py'); m=importlib.util.module_from_spec(spec); sys.modules['v542']=m; spec.loader.exec_module(m)
assert m.NEGATION_TO_POSITIVE['notcapableof']=='capable_of'
assert m.NEGATION_TO_POSITIVE['nothasproperty']=='has_property'
assert m.DEFAULT_WORKERS==20
print('V542 smoke test: PASS')
