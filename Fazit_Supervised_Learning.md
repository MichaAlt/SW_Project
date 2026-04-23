Die Qualität der Datensätze spielt eine sehr große Rolle, was vor allem bei den Datensätzen von Map5 deutlich zu beobachten ist. Während beim ersten Datensatz (training_data_map5_1.csv) sehr viel durch Drehen mit den Tasten A und D korrigiert wird, geschieht dies im zweiten Datensatz (training_data_map5_2.csv) deutlich seltener, was sich auch in den Modellen widerspiegelt.
Modelle des ersten Datensatzes bleiben häufig schon am Anfang der Strecke stecken, während Modelle des zweiten Datensatzes teilweise sogar die gesamte Strecke bewältigen können.

Mehr Datensätze bedeuten nicht automatisch bessere Modelle, da insbesondere zusätzliche, aber weniger konsistente die Modellleistung verschlechtern können.

Bei 150 Epochen und einer Batch-Size von 20 zeigt sich, dass der Optimizer SGD die Tendenz zu Overfitting reduziert, wodurch die KI die Strecke oft besser bewältigen kann als mit den Optimierern Adam oder AdamW. Letztere konvergieren schneller, neigen jedoch eher dazu, sich zu stark an die Trainingsdaten anzupassen, wodurch die Modelle in der Strecke stecken bleiben. Ähnliches gilt für die Nutzung von BatchNormalization-Layern nach den Hidden Layers, bei denen die Trainingsgenauigkeit zwar schneller konvergiert, dies jedoch nicht zu besseren Ergebnissen in der Simulation führt.
Zusätzlich konnte beobachtet werden, dass einige Modelle mit dem SGD-Optimizer die Strecken vollständig erfolgreich bewältigen konnten, während dies bei anderen Optimierern nicht zuverlässig gelang.

Bei 100 Epochen und einer Batch-Size von 120 tritt Overfitting insgesamt deutlich weniger auf, wodurch die Modelle stabiler und teilweise flüssiger laufen als die Modelle mit kleineren Batchgrößen. Dennoch gab es kein Modell, das die Strecke vollständig ohne Fehler, wie Umdrehen oder Kollisionen, bewältigen konnte.

Beim Training mit Supervised Learning zeigen sich Grenzen, da insbesondere die Qualität der Daten eine entscheidende Rolle spielt und bei schwierigeren Maps diese Daten nicht leicht in hoher Qualität zu erfassen sind. Zusätzlich lernt die KI nicht aus eigenen Fehlern und kann durch Overfitting zu schlechten Ergebnissen führen.
