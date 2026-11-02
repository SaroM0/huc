# Piloto exploratorio, 2026-11-02

Este registro resume pruebas de ingeniería, no un estudio de impacto en aprendizaje. Las
referencias esperadas son sintéticas y provisionales; el juez automático usa la misma familia
de modelos que el evaluador. Las muestras son pequeñas y no se calcularon intervalos.

## Casos sintéticos

| Prueba | Coincidencia con rangos previstos | Repeticiones | Observación |
| --- | --- | --- | --- |
| Ecuación lineal, inicial | 17/20 criterios | 2 por caso | Se penalizó dos veces un método alternativo; una vez se elogió una igualdad inválida. Ninguna ejecución fue señalada. |
| Ecuación lineal, primer ajuste | 14/20 criterios | 2 por caso | Cinco ejecuciones señaladas; el evaluador y su auditor también dieron por válido un paso inválido. |
| Ecuación lineal, segundo ajuste | 9/10 criterios | 1 por caso | Ambos caminos correctos recibieron el máximo; se señaló la respuesta con igualdades incompatibles, aunque el puntaje de razonamiento siguió siendo discutible. |
| Demostración de paridad, reservada | 8/10 criterios | 1 por caso | Aceptó dos pruebas válidas; discrepó de la referencia en dos puntuaciones de conclusión. |
| Geometría, reservada | 8/8 criterios | 1 por caso | Aceptó la ecuación equivalente expandida; tres ejecuciones fueron señaladas. |

Las dos ejecuciones lineales no son una comparación controlada: cambiaron tanto la rúbrica
como el prompt y se añadió una auditoría. El primer archivo conserva la definición del examen
en cada evaluación, pero no registró el hash del conjunto. Los archivos posteriores incluyen
hashes. Las tasas de un juez de IA se guardan como juicios exploratorios, no como etiquetas
independientes ni medidas demostradas de calidad.

La prueba incorrecta `2x = 4; x = 4` mostró errores correlacionados: tanto el generador como
el auditor llegaron a aprobar el paso. Por ello se agregó después una señal determinista y
limitada para igualdades lineales incompatibles. Puede producir alertas en soluciones con
casos o correcciones y no decide el puntaje ni verifica matemáticas generales.

En un ciclo posterior limitado a dos casos y ocho invocaciones lógicas previstas, el método
alternativo recibió puntaje completo. La cadena inválida recibió todavía puntaje parcial,
pero quedó marcada por la comprobación de igualdades. Se conservó un registro por caso; los
reintentos del proveedor y el costo en tokens no se midieron. El conjunto es demasiado pequeño
para estimar la eficacia de esta protección.

## Documentos reales

Se procesaron dos entregas de talleres diferentes, con seudónimos en los resultados. Los
originales y copias contienen imágenes y deben tratarse como material potencialmente
identificable. El primer taller se evaluó dos veces sobre la misma lectura: obtuvo 19/20 y
20/20 y en ambas se marcaron incertidumbres de OCR; una cita de evidencia no coincidió con
la transcripción. Una tercera evaluación, tras cambios en instrucciones y auditoría, obtuvo
19,5/20 y el auditor detectó, entre otros, un puntaje fuera de los niveles de la rúbrica.
Se añadió una alerta determinista para este último caso; no se redondea automáticamente.

En el segundo taller se comparó el enunciado con una rúbrica borrador. Se modificaron criterios
que exigían una construcción particular o un límite específico y se señaló una posible
degeneración geométrica del enunciado; no es una rúbrica docente oficial. La entrega disponible
consta de una página invertida. La lectura original omitió el diagrama; tras girar la página
y pedir una descripción explícita del dibujo se recuperaron elementos geométricos, pero ambas
lecturas tuvieron baja confianza y la solución siguió requiriendo inspección. El giro fue
manual para este experimento, no una capacidad automática del pipeline.

En la primera entrega se detectaron comentarios que pidieron aclarar pasos ya presentes o
describieron como requisito un método específico. Se ajustaron los prompts para distinguir
omisiones reales de mejoras opcionales. Una lectura incierta ahora también se advierte en
el feedback, sin afirmar que el aviso haga correcta la evaluación.

## Artefactos y límites

Los resultados detallados se encuentran en `data/research/`, que está excluido de Git. Cada
archivo de resultados de calificación posterior registra transcripción, criterios, propuestas,
revisión y prompts; las tasas automáticas no sustituyen la inspección semántica. No se puede
concluir de este piloto que el pipeline sea confiable para todas las soluciones ni que el
feedback cause una mejora en aprendizaje. El estudio de impacto requiere el diseño descrito
en [PROTOCOL.md](PROTOCOL.md), una evaluación independiente y datos de resultados de aprendizaje.
