# Evaluación sintética para investigación

Los casos son creados para probar el pipeline: no representan trabajos reales ni son una
medición de eficacia educativa. Los rangos son hipótesis iniciales, revisables según la
política del estudio; no equivalen a una rúbrica validada por docentes.

```bash
.venv/bin/python scripts/research_eval.py research/synthetic/math_cases.yaml \
  --repeats 3 --max-model-calls 60 --checkpoint data/research/synthetic-results.jsonl \
  --out data/research/synthetic-results.json
```

El comando llama al modelo configurado, incurre en consumo de API y conserva cada
resultado. Reporta coincidencia con rangos de puntaje y variabilidad entre repeticiones.
El conjunto usa texto sintético y no mide OCR. Sus cinco ejemplos no bastan para estimar
generalización, confiabilidad, calibración o impacto en aprendizaje.

`--case ID` limita el experimento a uno o más casos; `--max-model-calls` impide iniciar
si el número previsto de invocaciones lógicas pendientes supera el límite. No controla
reintentos del proveedor, tokens ni dinero. `--checkpoint` conserva cada caso terminado y
permite reanudar solo con el mismo conjunto, código y configuración; una caída a mitad de
un caso puede exigir repetir sus llamadas. No se sobrescribe un archivo final existente.
La salida cuenta también desacuerdos de puntaje con la referencia sintética que no fueron
señalados; esa cuenta no mide errores de feedback que la referencia no captura.

Opcionalmente `--judge-model NOMBRE` solicita a otro modelo una valoración estructurada
por dimensión, con explicación, almacenada en cada resultado. Consume llamadas adicionales;
es una evaluación exploratoria sujeta a sesgos y errores correlacionados, no una verdad de
referencia. Valide sus juicios de manera independiente antes de usar sus tasas como métricas.

Para estudiar el feedback, inspeccione cada `answer`, `focus`, `feedback` y la rúbrica
guardada en el resultado. Registre por separado, con ejemplos y razones, si el feedback:

1. Identifica un paso que realmente aparece en la respuesta.
2. Distingue un razonamiento válido, un error y una respuesta incompleta.
3. Explica matemáticamente el problema sin afirmar como cierto un paso incorrecto.
4. Reconoce aciertos y propone un siguiente paso útil y acorde con los criterios.
5. Coincide con los puntajes y no inventa ni impone un método único.

Una revisión independiente por un investigador, o un evaluador automático validado para
ese propósito, puede producir esas etiquetas. Una nota o confianza autodeclarada por el
mismo modelo no demuestra corrección. Para mediciones comparables, fijar el conjunto,
versiones y configuración antes de comparar modelos; conservar todas las repeticiones y
documentar los cambios de criterios y fuentes de desacuerdo. Para estudiar impacto en
aprendizaje se necesitaría además un diseño de investigación y mediciones de aprendizaje;
la calidad de generación por sí sola no lo demuestra.
