# Perfil longitudinal del estudiante

> **Estado: implementado (fases a y b); falta validar.** Las dimensiones y descriptores son un
> borrador para revisión del equipo docente. Antes de interpretar los cambios hay que correr la
> validación de la fase c (piso de ruido y calibración), que aún no está hecha.

## Uso rápido

```bash
huc exam-load examples/taller_s14_m1.yaml   # el examen necesita `sequence: {week, module}`
huc ingest s14-m1 data/Semana_14_Modulo_M1/  # como siempre
huc profile-run s14-m1                       # califica cada entrega en las 7 dimensiones
huc profile EST-017                          # línea base, niveles por entrega, tendencia
huc profile-export --out profiles.csv        # una fila por estudiante × entrega × dimensión
```

`profile-run` no necesita que la entrega esté calificada y no vuelve a evaluar las entregas ya
evaluadas con las dimensiones actuales (`--force` las repite). API: `POST
/exams/{id}/profile`, `GET /students`, `GET /students/{id}/profile`, `GET /profiles/export.csv`.

## Qué resuelve

Hoy el pipeline califica cada entrega por separado: puntaje por criterio, feedback y una
etiqueta de verificación. No hay forma de responder *¿este estudiante escribe mejores
demostraciones que al comienzo del semestre?*

El perfil agrega esa vista. Evalúa cada entrega de cada estudiante en un conjunto fijo de
**dimensiones transversales**, guarda una **línea base** (su primer trabajo) y muestra, dimensión
por dimensión, si el desempeño observado mejora, se mantiene o retrocede.

## Principios

| Principio | Qué significa |
|---|---|
| Dimensiones comunes a todo el curso | Todos los estudiantes se evalúan en las mismas dimensiones, definidas una vez en el perfil del curso (`profiles/mat0004.yaml`). |
| Evaluación individual | Cada entrega de cada estudiante recibe un nivel por dimensión, con evidencia literal de su respuesta. |
| Línea base fija | El primer trabajo entregado por el estudiante. No se mueve durante el semestre. |
| Evaluación a ciegas | El agente perfilador **no ve** el perfil previo ni los puntajes de la entrega, para no anclarse ni repetir la nota. |
| Comparación determinista | La mejora o el retroceso los calcula el código a partir de los niveles guardados, no el LLM. |
| Descriptores absolutos | Los niveles describen la calidad del razonamiento escrito, no la dificultad del tema, así un nivel 3 en S01 equivale a un 3 en S14. |
| Sin afirmaciones causales | El reporte habla de *desempeño observado*, no de *aprendizaje* (ver `research/PROTOCOL.md`). |

## Flujo

```
 entrega escaneada
        |
        v
 INGESTION  ->  Submission (respuestas transcritas)
        |
        v
 GRADING    ->  Grade (puntajes, feedback)              <- sin cambios
        |
        v
 +-----------------------------------------------------------+
 | PROFILING GRAPH  (graphs/profiling.py, nuevo)             |
 |                                                           |
 |  load: Submission + Exam (preguntas, dimensiones)         |
 |        -- NO carga Grade ni observaciones previas --      |
 |    |                                                      |
 |    v                                                      |
 |  rate_dimensions: 1 llamada Haiku por entrega             |
 |    -> por dimensión: nivel 1-4 o no_observable,           |
 |       evidencia literal, justificación, confianza         |
 |    |                                                      |
 |    v                                                      |
 |  save: DimensionObservation (append-only)                 |
 +-----------------------------------------------------------+
        |
        v
 compute_profile(student_id)      determinista, sin LLM
   -> línea base, serie, delta y estado por dimensión
        |
        v
 huc profile <student_id>  /  huc profile-export  (CSV)
```

El grafo de perfilamiento es independiente del de calificación: se puede correr sobre entregas
ya calificadas (backfill) y repetir si cambian las dimensiones, sin recalificar.

## Las dimensiones

Salen del material del curso: los objetivos del taller (`presentacion2026`), la guía de
corrección para ayudantes, las «Orientaciones generales» de la retroalimentación de S09 L5 y los
criterios de los libretos de Cálculo, Álgebra y Geometría. Cada una se puede observar en casi
cualquier problema de cualquier semana.

| # | id | Dimensión | Pregunta que responde |
|---|---|---|---|
| 1 | `justificacion` | Justificación de los pasos | ¿Dice por qué vale cada paso? |
| 2 | `estructura_logica` | Estructura lógica | ¿El argumento, como un todo, demuestra lo pedido? |
| 3 | `dominio_conceptual` | Dominio conceptual | ¿Usa las definiciones y resultados correctos? |
| 4 | `generalidad` | Generalidad y completitud | ¿Cubre todos los casos y todo lo pedido? |
| 5 | `precision_simbolica` | Precisión simbólica | ¿Sus letras, cuantificadores y símbolos son precisos? |
| 6 | `comunicacion` | Comunicación escrita | ¿Un lector puede seguir la solución? |
| 7 | `procedimiento` | Corrección procedimental | ¿Los cálculos están bien? |

La definición completa, con descriptores, está en `profiles/mat0004.yaml` (sección `profiling`).

### Escala

| Nivel | Nombre |
|---|---|
| 1 | incipiente |
| 2 | en desarrollo |
| 3 | logrado |
| 4 | destacado |
| — | `no_observable`: la entrega no permite verla (respuesta en blanco, o ningún problema la exige). **No es un nivel bajo** y no entra en la comparación. |

### Descriptores

Cada dimensión define su frontera (`not_this`), así un mismo error no se penaliza en tres
dimensiones a la vez.

**1. Justificación de los pasos** (`justificacion`)
Cada paso se apoya en un axioma, definición, teorema o hipótesis, nombrado o claramente
identificable, y respeta las restricciones del enunciado («a partir de los axiomas», «usando la
propiedad arquimediana»).
*No es:* el orden de la redacción (→ comunicación) ni el método global (→ estructura lógica).

| Nivel | Descriptor |
|---|---|
| 1 | Afirma resultados o encadena fórmulas sin indicar por qué valen. |
| 2 | Justifica algunos pasos; deja sin justificar pasos clave o usa resultados no permitidos. |
| 3 | Justifica los pasos esenciales; faltan detalles menores. |
| 4 | Cada paso relevante indica qué usa y por qué aplica en ese punto. |

**2. Estructura lógica** (`estructura_logica`)
Distingue hipótesis de tesis, usa un método válido (directo, contrapositivo, contradicción,
inducción, doble inclusión), prueba ambas direcciones de un ⇔ y no asume lo que debe demostrar.
*No es:* el nombre del método. Llamar «por contradicción» a una demostración por contrapositivo
no se penaliza (criterio del libreto de Álgebra).

| Nivel | Descriptor |
|---|---|
| 1 | Argumento circular, demuestra la recíproca o no hay una línea de razonamiento identificable. |
| 2 | Hay una estrategia, pero falta una dirección, el caso base o el cierre del argumento. |
| 3 | Estructura válida y completa, con pequeñas imprecisiones. |
| 4 | Estructura válida y explícita: anuncia método y objetivo, y cierra cada parte. |

**3. Dominio conceptual** (`dominio_conceptual`)
Usa las definiciones y los resultados correctos, no confunde objetos (función y valor, inversa e
imagen inversa, condición necesaria y suficiente) y usa la definición cuando se pide.
*No es:* omitir la cita de una definición que sí usó bien (→ justificación).

| Nivel | Descriptor |
|---|---|
| 1 | Usa una definición o un resultado incorrecto, o confunde los objetos centrales del problema. |
| 2 | Conoce las definiciones, pero las aplica mal en un punto importante o usa un atajo no permitido en vez de la definición pedida. |
| 3 | Usa correctamente las definiciones y resultados, con una imprecisión menor. |
| 4 | Escribe y aplica con exactitud las definiciones relevantes y distingue con claridad los objetos. |

**4. Generalidad y completitud** (`generalidad`)
No generaliza desde casos particulares, cubre todos los casos y configuraciones (también las que
el dibujo no muestra), trata los casos especiales y responde todo lo pedido.
*No es:* omitir una dirección de un ⇔ (→ estructura lógica).

| Nivel | Descriptor |
|---|---|
| 1 | Concluye a partir de ejemplos o de un dibujo particular, o deja sin responder buena parte de lo pedido. |
| 2 | El caso general está, pero omite casos especiales relevantes (división por cero, signos, configuraciones) o una parte de lo pedido. |
| 3 | Cubre el caso general y casi todos los casos especiales. |
| 4 | Identifica y trata explícitamente todos los casos y responde todo lo pedido. |

**5. Precisión simbólica** (`precision_simbolica`)
Define cada letra antes de usarla, respeta el orden de los cuantificadores, niega correctamente y
no reutiliza una variable con dos significados.
*No es:* la legibilidad de la letra manuscrita ni lo que el OCR no pudo leer (`[illegible]` no
cuenta contra el estudiante).

| Nivel | Descriptor |
|---|---|
| 1 | Variables sin introducir o con doble significado, o cuantificadores o negaciones que cambian el sentido. |
| 2 | La notación es en general correcta, con imprecisiones que obligan a adivinar en algún punto. |
| 3 | Notación correcta y variables introducidas; deslices menores sin consecuencias. |
| 4 | Notación precisa y consistente; cada objeto se introduce con su dominio. |

**6. Comunicación escrita** (`comunicacion`)
Desarrollo ordenado, símbolos acompañados de frases y conectores, modelo o dibujo descrito antes
de calcular, y conclusión explícita. Corresponde al objetivo principal 3 del taller.
*No es:* la validez del argumento (→ estructura lógica) ni la ortografía.

| Nivel | Descriptor |
|---|---|
| 1 | Fórmulas sueltas sin texto; no se puede seguir el razonamiento. |
| 2 | Se puede seguir con esfuerzo; faltan conectores o conclusión, o el orden es irregular. |
| 3 | Ordenada y legible, con frases que conectan los pasos y una conclusión. |
| 4 | Se lee como un texto matemático: plantea, desarrolla y concluye explícitamente lo pedido. |

**7. Corrección procedimental** (`procedimiento`)
Álgebra, desigualdades y cálculos sin errores, con una estrategia adecuada.
*No es:* elegir una estrategia equivocada (→ estructura lógica o dominio conceptual).

| Nivel | Descriptor |
|---|---|
| 1 | Errores de cálculo que invalidan el resultado en varios puntos. |
| 2 | Un error que afecta el resultado, o varios deslices menores. |
| 3 | Cálculos correctos con un desliz menor que no afecta la conclusión. |
| 4 | Cálculos correctos y verificados. |

### Lo que no se mide (y por qué)

- **Decidir V/F, o entre demostración y contraejemplo**: aparece solo en unas 8 de 13 semanas y
  ya queda cubierto por las dimensiones 2 y 3.
- **Aprender a estudiar, autorregulación, aprender del error**: son objetivos del taller, pero no se
  ven en un escaneo. Podrían medirse con las etapas `REFLECTION` y `FINAL_SOLUTION` de
  `WorkshopSession`.
- **Competencias de un solo curso** (ε-N, regla y compás): no aparecen todas las semanas. Ese
  desempeño lo mide la nota por `subject`.

## Configuración

Las dimensiones viven en el perfil del curso, que se copia dentro de cada `Exam` al cargarlo.
Cada observación queda así ligada a la versión exacta de las dimensiones con que se evaluó.

```yaml
# profiles/mat0004.yaml
profiling:
  version: 1               # cambiar las dimensiones -> subir la versión
  baseline: first_submission
  min_change: 1            # diferencia mínima, en niveles, para hablar de cambio
  levels:
    "1": incipiente
    "2": en desarrollo
    "3": logrado
    "4": destacado
  dimensions:
    - id: justificacion
      name: Justificación de los pasos
      description: >
        Cada paso se apoya en un axioma, definición, teorema o hipótesis, nombrado o
        claramente identificable, y respeta las restricciones del enunciado.
      not_this: El orden de la redacción (comunicacion) ni el método global (estructura_logica).
      anchors:
        "1": Afirma resultados o encadena fórmulas sin indicar por qué valen.
        "2": Justifica algunos pasos; deja sin justificar pasos clave o usa resultados no permitidos.
        "3": Justifica los pasos esenciales; faltan detalles menores.
        "4": Cada paso relevante indica qué usa y por qué aplica en ese punto.
      error_types: [justificacion]   # señal derivada de las calificaciones, sin LLM
    # ... las otras 6 dimensiones con la misma forma
```

`error_types` enlaza cada dimensión con los tipos de error que el calificador ya registra
(`conceptual`, `justificacion`, `calculo`, `notacion`, `incompleta`). Así se obtiene una segunda
señal, gratuita y reproducible, para contrastar con la del perfilador.

## Orden de las entregas

`Exam` necesita un campo de orden, que se deduce de la carpeta `Semana_XX_Modulo_YY`:

```yaml
sequence: { week: 14, module: M1 }
```

Cada semana tiene varios talleres distintos (L5, L6, M1, M2, M3, W5, W6), así que el orden es
**semana → día (L → M → W) → módulo**. La primera entrega de un estudiante es la menor en ese
orden.

## Modelo de datos

```
DimensionObservation        se guarda; inmutable (como grade_history)
  student_id, exam_id, submission_id
  sequence                  orden de la entrega
  dimension_id
  level: 1-4 | None         None = no_observable
  evidence: [str]           citas literales de la respuesta
  rationale: str
  confidence: 0-1
  profiling_version, dimensions_hash, model, created_at

StudentProfile              no se guarda; se calcula de las observaciones
  student_id
  dimensions: [
    { dimension_id, baseline_level, baseline_exam,
      series: [(exam, level)], current_level, delta, status }
  ]
```

Guardar observaciones y **calcular** el perfil permite cambiar la regla de comparación (umbral,
qué se muestra) y recalcular sin volver a llamar al modelo. Las observaciones de distinta
`profiling_version` no se comparan entre sí.

## Línea base y comparación

1. **Línea base** = nivel de la dimensión en la **primera entrega** del estudiante.
2. Si la dimensión es `no_observable` en esa entrega, la línea base de **esa dimensión** se fija en
   la primera entrega en que sí sea observable, y el reporte lo indica (*«línea base: S03 M2»*).
   Así no se compara contra un nivel inventado.
3. Para cada entrega: `delta = nivel − línea_base`. El **nivel actual** es el de la última
   entrega en que la dimensión fue observable; el reporte indica de qué entrega sale.
4. Estado de la dimensión:

| Estado | Regla |
|---|---|
| `mejora` | `delta ≥ min_change` |
| `retroceso` | `delta ≤ −min_change` |
| `estable` | `|delta| < min_change` |
| `solo_linea_base` | observada una sola vez: aún no hay con qué comparar |
| `sin_linea_base` | la dimensión aún no ha sido observable |

Además, cada nivel lleva una **marca de baja confianza** (`?` en la CLI, `low_confidence` en
JSON y CSV) cuando su `confidence` está bajo el umbral de verificación del perfil, o cuando el
código detectó un problema en la salida del modelo: una cita que no está en la transcripción,
un nivel fuera de la escala o una dimensión sin calificar. La confianza nunca supera la
confianza de la lectura OCR de la entrega.

### Ejemplo de reporte

Salida real de `huc profile`, con niveles sintéticos:

```
Student EST-017  baseline: S03 M3  submissions: 3  profiling v1

dimension               S03 M3   S08 L5   S14 M3  delta  status
justificacion                2        2       3?     +1  mejora [low confidence]
estructura_logica            2        3       3?     +1  mejora [low confidence]
dominio_conceptual           3        3       3?     +0  estable [low confidence]
generalidad                  1        2       2?     +1  mejora [low confidence]
precision_simbolica          3        3       2?     -1  retroceso [low confidence]
comunicacion                 2        2       2?     +0  estable [low confidence]
procedimiento                -        3       3?     +0  estable (baseline S08 L5) [low confidence]

-: not observable   ?: low confidence or failed check
```

`huc profile-export` produce lo mismo en CSV (una fila por estudiante × entrega × dimensión) para
análisis. La columna `error_count` cuenta, en la calificación de esa entrega, los `error_type` del
calificador vinculados a la dimensión (`error_types` en el YAML). Es una segunda señal, calculada
sin LLM, para contrastar con el perfilador.

## Por qué el perfilador evalúa a ciegas

- **Sin el perfil previo**: si el modelo sabe que el estudiante era nivel 2, tiende a confirmarlo
  (sesgo de anclaje). Cada entrega se evalúa como si fuera la única.
- **Sin los puntajes**: si el modelo ve la nota, las 7 dimensiones terminan siendo la nota en 7
  columnas y no aportan información nueva. Que las dimensiones **no** estén perfectamente
  correlacionadas con la nota es en sí un chequeo de validez.
- Recibe solo: enunciados, respuestas transcritas, dimensiones con sus descriptores y las reglas
  de lectura del curso (`[illegible]` no es error del estudiante).

## Validación antes de confiar en los deltas

1. **Piso de ruido**: perfilar la misma entrega 3–5 veces y medir cuánto varía el nivel. Si varía
   ±1, `min_change: 1` no basta para hablar de cambio y hay que subir el umbral o promediar
   repeticiones.
2. **Calibración**: un ayudante asigna niveles a 20–30 entregas sin ver los del modelo; se mide el
   acuerdo por dimensión. Una dimensión con acuerdo bajo se reescribe o se elimina.
3. **Par de control S08 L5 / S09 L5**: S09 L5 retoma los problemas de S08 L5 tras la
   retroalimentación. Mismo estudiante, mismo problema: es el caso más limpio para ver si el
   sistema detecta un cambio en la redacción.

## Privacidad

Los `student_id` salen hoy del nombre del archivo (nombre real). Los reportes y exportaciones de
investigación deben usar seudónimos (`EST-017`); la tabla de correspondencia queda fuera de git,
junto al resto de los datos personales en `data/`.

## Implementación

| Archivo | Qué contiene |
|---|---|
| `src/huc/domain/profile.py` | `Dimension`, `ProfilingPolicy` (sección `profiling` del perfil), validación y `dimensions_hash` |
| `src/huc/domain/models.py` | `ExamSequence` (semana y módulo) en `Exam.sequence` |
| `src/huc/domain/student_profile.py` | Salida del modelo, `SubmissionProfile` (registro guardado), `compute_profile` (línea base y tendencia, determinista) |
| `src/huc/prompts/profiling.py` | Prompt del perfilador: curso + dimensiones (caché para todo el curso), preguntas (caché por examen), respuestas |
| `src/huc/graphs/profiling.py` | Grafo `load → rate → save`, controles sobre la salida del modelo, conteo de `error_types` |
| `src/huc/storage/repository.py` | `save_submission_profile` / `list_submission_profiles` (append-only en `data/student_profiles/`) |
| `src/huc/services.py`, `cli.py`, `api/app.py` | `profile_exam`, `student_profile`, `export_profiles_csv` y sus comandos y endpoints |
| `profiles/mat0004.yaml` | Las 7 dimensiones con sus descriptores |
| `tests/test_student_profile.py` | Configuración, evaluación a ciegas, reglas de comparación, CLI, CSV y API |

`huc rubric-draft` agrega `sequence` a los YAML que genera. Los exámenes ya cargados sin
`sequence` siguen funcionando, pero se ordenan por fecha de ingesta y el perfil lo advierte.

| Fase | Contenido | Estado |
|---|---|---|
| a | Sección `profiling`, modelos, `sequence` en `Exam`, señal derivada de `error_types` | Hecha |
| b | Grafo perfilador con Haiku (`HUC_PROFILER_MODEL`, por defecto `claude-haiku-5-5`), CLI y API | Hecha (probada con modelo falso; falta una corrida real) |
| c | Scripts de piso de ruido y calibración; prueba con el par S08 L5 / S09 L5 | Pendiente |

## Decisiones pendientes

- Confirmar las 7 dimensiones y sus descriptores con el equipo docente.
- Varias entregas del mismo estudiante para el mismo examen (p. ej. un escaneo subido en dos
  archivos) se tratan como entregas separadas; el perfil lo advierte. Unirlas es tarea de la
  ingesta.
- Conseguir los documentos «lista de errores comunes» y «Orientaciones para hacer
  demostraciones» que citan los libretos y no están en `data/course/`; pueden afinar los
  descriptores.
