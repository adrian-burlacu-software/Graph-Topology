
# V456 — Teacher interface fix

The optional SmolLM2 fallback now uses the actual inherited interface:

```python
teacher.answer(question, context)
```

instead of the nonexistent:

```python
teacher.generate(...)
```

Run without teacher:

```powershell
python .\research\v456\v456_teacher_interface_fix.py
```

Run with fallback:

```powershell
python .\research\v456\v456_teacher_interface_fix.py `
  --teacher ".\llm\SmolLM2-1.7B-Instruct"
```

Persistent memory:

```text
C:\Users\adria\Desktop\dev\Graph-Topology\results\assistant_semantic_net.sqlite
```
