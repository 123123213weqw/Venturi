from pathlib import Path
s=Path('/tmp/venturi-token16-observe.py').read_text().replace('venturi-token16-study-v1','venturi-token16-staged-study-v5').replace("['candidate-t16']","['candidate-staged16']").replace("'3vdt3dotILi16ELi2ELi8E','3vqf9tiled_dotILi16ELi2ELi8E'", "'3vdt12staged16_dotILi2ELi8E','3vqf12staged16_dotILi2ELi8E'").replace('operator-observation.json','staged-operator-observation.json')
exec(compile(s,__file__,'exec'))
