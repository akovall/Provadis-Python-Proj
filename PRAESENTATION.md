# Raster → SVG: Material für die Projektpräsentation

Stichpunkte für die Präsentation, gegliedert nach Themen. Zu jedem Algorithmus steht dabei, welches **Problem** er löst, **wie** er es löst und was dabei **herauskam** (Messwerte aus `changes.txt`, README und Commits).

---

## 1. Projektüberblick

- **Ziel:** Rasterbilder (`.jpg`, `.jpeg`, `.png`) automatisch in saubere Vektorgrafiken (`.svg`) umwandeln: glatte Kurven, wenige Punkte, echte Ebenen, transparenter Hintergrund.
- **Zwei Oberflächen:**
  - Kommandozeile (`main.py`) mit Presets und Optionen
  - Web-App (FastAPI + reines HTML/JS) mit Live-Schiebereglern, Vorher/Nachher-Vergleich, Ebenen-Panel, Galerie und PNG-Export
- **Technologie:** Python 3.10+, OpenCV, NumPy, FastAPI/Uvicorn, `resvg-py` (SVG-Renderer zum Messen), `onnxruntime` (KI-Upscaling). Frontend ohne Framework.
- **Umfang:** ca. 3 500 Zeilen (≈ 2 300 Python, ≈ 1 300 HTML/CSS/JS), 13 Pull Requests.
- **Team:** Andrii Koval und Anna Smahliuk. Gearbeitet wurde mit Feature-Branches und Pull Requests auf GitHub.
- **Kernidee:** Am Anfang stand die fertige Bibliothek **vtracer**. Sie wurde schrittweise durch eine **eigene Pipeline** ersetzt (eigene Ebenenbildung, eigene Kurvenanpassung, eigene Formerkennung, eigene Konturlinien).

---

## 2. Architektur: die Pipeline

```
Bild ──► preprocess.py ──► layers.py ──► shapes.py ──► curves.py ──► SVG
          (Freistellen,     (Ebenen,      (Ellipse/     (Bézier-
           Farben, Rauschen) Stapeln)      Kreis/Rechteck) Anpassung)
                               │
                               └──► strokes.py (Konturlinien als Bänder, ganz oben)

quality.py   : SVG zurück in Pixel rendern und mit dem Original vergleichen
presets.py   : Bildtyp erkennen (Logo / Cartoon / Foto)
upscale.py   : optionales KI-Upscaling (Real-ESRGAN, ONNX)
vectorize.py : setzt alles zusammen und schreibt das SVG
app.py       : Web-API mit Caching   ·   gallery.py: Galerie im Hintergrund
```

| Modul | Aufgabe |
|---|---|
| `preprocess.py` | Hintergrund entfernen, Arbeitsgröße, Glätten, Kanten schärfen, Farbpalette, Pixel zuordnen |
| `layers.py` | Flächen bilden, säubern, von groß nach klein stapeln, Ebenen zusammenfassen und reduzieren |
| `shapes.py` | Ebenen, die Ellipsen, Kreise oder Rechtecke sind, als echte SVG-Formen ausgeben |
| `curves.py` | Kanten mit möglichst wenigen kubischen Bézier-Kurven anpassen (Schneider-Algorithmus) |
| `strokes.py` | Dunkle, dünne Linien finden und als glattes Band mit variabler Dicke neu zeichnen |
| `quality.py`, `gallery.py` | Qualität messen (ΔE bzw. Fehler 0–255) |
| `presets.py` | Bildtyp automatisch erkennen |
| `upscale.py` | 4×-Upscaling kleiner Zeichnungen mit einem neuronalen Netz |
| `text.py` | Schriftzüge finden und jede Kopie eines Buchstabens aus einer gemeinsamen Form zeichnen |

### Der Ablauf in Worten: was passiert, warum und womit

Zuerst schaut das Programm sich das Bild an und entscheidet, ob es ein Foto, ein Logo oder eine Zeichnung ist. Davon hängen die Einstellungen ab: wie viele Farben bleiben, wie stark geglättet wird und welche kleinen Flecken wegfallen. Dafür misst **OpenCV**, welcher Anteil des Bildes aus ruhigen, gleichmäßigen Flächen besteht, und **NumPy** zählt, wie viele Farben fast die ganze Fläche abdecken. Ist das Bild klein, kann es vorher mit dem neuronalen Netz Real-ESRGAN vergrößert werden, das **onnxruntime** direkt auf der CPU ausführt. In einem kleinen Bild sind Buchstaben und dünne Linien nur ein paar Pixel breit, und ohne Vergrößerung werden ihre Kanten wellig.

Dann wird das Bild mit **OpenCV** bereinigt. Der Hintergrund wird über die Farben am Bildrand erkannt und durchsichtig gemacht. Rauschen und Texturen glättet der Mean-Shift-Filter, und die Grenzen zwischen Farben werden geschärft, damit an den Kanten keine überflüssigen Streifen entstehen. Danach wählt **k-means aus OpenCV** die 10 bis 20 wichtigsten Farben, und jedes Pixel bekommt die nächstliegende davon. Die Farben werden im Lab-Farbraum verglichen, weil dort der Abstand zwischen zwei Farben ungefähr dem entspricht, was ein Mensch sieht. Das ist nötig, weil ein SVG klare Flächen in einer Farbe braucht und nicht Millionen von Farbtönen.

Anschließend wird jede Farbe zu einer eigenen Ebene. Die Ebenen werden übereinandergelegt: große unten, kleine oben. So lässt sich ein Gesicht als eine ganze Form zeichnen, und das Auge liegt einfach darauf, ohne Loch im Gesicht. Dadurch gibt es weniger Formen, und das SVG lässt sich bequem in einem Editor bearbeiten. Die Flächen und ihre Ränder findet **OpenCV** (`connectedComponents`, `findContours`). Für jede Fläche wird geprüft, ob sie ein Kreis, eine Ellipse oder ein Rechteck ist. Wenn ja, schreibt das Programm ein echtes `<circle>` oder `<rect>` in die Datei statt einer Kurve aus hundert Punkten. Dünne dunkle Umrisse werden auf ihre Mittellinie reduziert und als Linie mit passender Dicke gespeichert. Gleiche Buchstaben im Text werden zu einer gemeinsamen Form gemittelt, deshalb sieht jedes „a“ gleich sauber aus. Diese drei Algorithmen (Formen, Umrisse, Buchstaben) haben wir selbst mit **NumPy** geschrieben.

Danach werden die Pixelränder, die aus Treppenstufen bestehen, in glatte **Bézierkurven** umgewandelt. Das Programm findet zuerst die echten Ecken, damit sie nicht abgerundet werden, glättet die Abschnitte dazwischen und passt die Kurven mit dem Verfahren von Schneider an. Auch dieser Algorithmus ist selbst mit NumPy geschrieben und nutzt alle Prozessorkerne. So werden aus Tausenden Punkten wenige Kurven: Die Datei ist klein, und der Rand bleibt bei jeder Vergrößerung glatt.

Am Ende zeichnet das Programm das SVG mit OpenCV wieder als Bild und vergleicht es über den Farbabstand ΔE mit dem Original. Daraus entsteht eine einzige Zahl, die zeigt, wie ähnlich das Ergebnis dem Ausgangsbild ist. Die Weboberfläche läuft mit **FastAPI** und **uvicorn**. Sie speichert Zwischenergebnisse, deshalb wird beim Verschieben eines Reglers nur der geänderte Schritt neu berechnet. Die Galerie misst die Qualität zusätzlich mit **resvg**, das SVG genau wie ein Browser zeichnet; den PNG-Export erledigt der Browser selbst.

### Bibliotheken und welche Probleme sie lösen

- **OpenCV** (`opencv-python`) übernimmt die Bildverarbeitung in allen Schritten:
  - `imread` / `imdecode` öffnen PNG- und JPG-Dateien, auch Uploads aus dem Browser, und `imencode` speichert die PNGs der Galerie.
  - `Sobel` misst Helligkeitsänderungen und unterscheidet so Fotos von Zeichnungen mit flachen Flächen (Presets).
  - `kmeans` findet die Randfarben, und `connectedComponents` behält nur die Flächen dieser Farben, die den Rand berühren. So wird ein weißer Fleck im Objekt nicht auch durchsichtig.
  - `pyrMeanShiftFiltering` mittelt ähnliche Nachbarpixel, ohne die Grenzen zu verwischen. Ohne diesen Filter würden JPEG-Rauschen und Texturen zu Tausenden winziger Formen.
  - `cvtColor` rechnet in den Lab-Farbraum um, und `kmeans` wählt die Palette: eine schnelle C++-Implementierung statt einer langsamen Python-Schleife.
  - `connectedComponentsWithStats` und `findContours` finden die Flächen und ihre Ränder, `dilate` / `morphologyEx` schließen Lücken zwischen Ebenen, und `moments`, `fitEllipse` und `minAreaRect` liefern Mittelpunkt, Größe und Winkel für den Test auf Kreise, Ellipsen und Rechtecke.
  - `distanceTransform` liefert für jeden Punkt den Abstand zum Rand, also die Dicke einer Konturlinie an dieser Stelle.
  - `fillPoly` und `ellipse` zeichnen das SVG wieder in Pixel, für den Live-Ähnlichkeitswert.
- **NumPy** verarbeitet das ganze Bild als Array, ohne Schleifen über Pixel. Alles, was OpenCV nicht bietet, ist damit geschrieben: das Zhang-Suen-Skelett, die Bézier-Anpassung nach Schneider, die Eckenerkennung, das Mitteln der Buchstaben und die ΔE-Berechnung.
- **onnxruntime** führt das Netz Real-ESRGAN auf einer normalen CPU aus, ohne gigabytegroßes PyTorch und ohne Grafikkarte. Ohne Upscaling werden Buchstaben und Linien in kleinen Bildern wellig. Ist die Bibliothek nicht installiert, wird das KI-Upscaling einfach abgeschaltet.
- **resvg** (`resvg-py`) zeichnet ein SVG genau wie ein Browser in Pixel. Die Galerie misst damit die Qualität, denn OpenCV kann kein SVG lesen.
- **FastAPI** macht aus Python-Funktionen eine Web-API (Upload, Vektorisieren, Galerie). Die **pydantic**-Modelle prüfen die Eingaben, deshalb bekommt eine Anfrage mit `colors=500` sofort einen Fehler, statt die Berechnung zu stören.
- **uvicorn** ist der Webserver, der die FastAPI-App startet und dem Browser antwortet.
- **python-multipart** ermöglicht FastAPI das Empfangen hochgeladener Dateien (`UploadFile`). Ohne diese Bibliothek funktioniert der Bild-Upload nicht.
- **Python-Standardbibliothek:** `concurrent.futures` verteilt die Kurvenanpassung auf alle Prozessorkerne, `threading` erstellt die Galerie im Hintergrund, ohne den Nutzer zu bremsen, und `argparse` liest die Kommandozeilenoptionen in `main.py`.
- **Im Browser** kommt die Seite ohne Framework aus: Der PNG-Export zeichnet das SVG auf ein `<canvas>` (`static/export.js`).

### Alle Algorithmen in je drei Sätzen

In der Reihenfolge der Pipeline:

- **Hintergrund freistellen** (`preprocess.py`): Hat das Bild keinen echten Alphakanal, werden zuerst die häufigsten Randfarben mit k-means bestimmt, so wird auch ein gemaltes Schachbrettmuster erkannt. Alle Pixel, die einer Randfarbe ähnlich sind (ΔE unter 12), sind Kandidaten für den Hintergrund. Hintergrund wird aber nur, was zusammenhängend den Bildrand berührt, deshalb bleiben gleichfarbige Flächen innerhalb des Objekts erhalten.
- **Arbeitsgröße** (`preprocess.py`): Jedes Bild wird auf etwa 1024 px gebracht: Große Bilder werden verkleinert, kleine bis zu 4× vergrößert. Beim Vergrößern entfernt ein leichter Gauß-Weichzeichner die Pixeltreppen, damit dünne Linien als glatte Kanten erhalten bleiben. Das fertige SVG behält über die `viewBox` trotzdem die Originalgröße.
- **Mean-Shift-Filter** (`preprocess.py`): Der Filter ersetzt jedes Pixel durch den Mittelwert ähnlicher Farben in seiner Umgebung (Radius 8 px, Farbbereich 18). Dadurch verschwinden Texturen und JPEG-Rauschen, während Kanten zwischen deutlich verschiedenen Farben scharf bleiben. Die Flächen werden so flacher und zerfallen später nicht in viele kleine Farbinseln.
- **Toggle-Filter (Kanten schärfen)** (`preprocess.py`): Jedes Pixel übernimmt die dunkelste oder die hellste Farbe aus seiner 3×3-Nachbarschaft, je nachdem, welche seiner eigenen Helligkeit näher ist. Ein weicher, kantengeglätteter Übergang wird so zu einer scharfen Stufe und bildet kein eigenes Band aus Zwischenfarben. Flache Bereiche bleiben unverändert, und es laufen nur 2 Durchgänge, weil mehr Durchgänge die Kanten zackig machen.
- **Farbpalette** (`preprocess.py`): Die Farben des Objekts werden in den Lab-Farbraum umgerechnet, weil Abstände dort (ΔE) dem menschlichen Farbempfinden entsprechen. k-means++ sucht auf einer Stichprobe von 20 000 Pixeln die gewünschte Anzahl an Farben, mit festem Seed, damit dasselbe Bild immer dasselbe Ergebnis liefert. Danach wird immer das ähnlichste Farbpaar gewichtet zusammengelegt, bis alle Farben mindestens den eingestellten ΔE-Abstand haben.
- **Pixel zuordnen und Mehrheitsfilter** (`preprocess.py`): Das Bild wird linear auf die doppelte Größe gebracht, und jedes Pixel bekommt die nächstgelegene Palettenfarbe. Danach übernimmt jedes Pixel die häufigste Farbe seiner 5×5-Umgebung. Das entfernt Treppenstufen und einzelne Sprenkel, bevor die Kanten nachgezeichnet werden.
- **Sprenkel und Säume säubern** (`layers.py`): Kleine Flächen gehen an die farblich ähnlichste Nachbarfläche, damit zum Beispiel ein Stück einer blassen Linie mit der Linie verbunden bleibt. Ein dünner Streifen zwischen zwei viel größeren Flächen, dessen Farbe im Lab-Raum zwischen deren Farben liegt, gilt als Anti-Aliasing-Saum. Jedes Saumpixel geht an die Seite, der es farblich näher ist, sodass die Kante genau in der Mitte liegt.
- **Ebenen stapeln** (`layers.py`): Jede zusammenhängende Fläche einer Farbe wird zu einer Ebene, und die Ebenen werden von der größten unten bis zur kleinsten oben gestapelt. Löcher und Einkerbungen unter den oberen Ebenen werden mit morphologischem Schließen aufgefüllt, deshalb bleibt ein Gesicht eine ganze Form mit den Augen obendrauf. Jede Ebene reicht zusätzlich 4 px unter die Ebenen darüber, damit nach dem Glätten keine Lücken aufgehen.
- **Ebenen zusammenfassen** (`layers.py`): Formen derselben Farbe kommen in eine gemeinsame Ebene, wenn keine dazwischen gezeichnete Ebene sie überlappt. Das Bild sieht danach genau gleich aus, hat aber viel weniger Ebenen. Beim Pilz werden so aus 145 Formen 18 Ebenen.
- **Ebenen reduzieren** (`layers.py`): Bei einer Zielanzahl werden die unwichtigsten Ebenen entfernt, die Kosten werden als Fläche × Farbdifferenz (ΔE) berechnet. Dabei verschmilzt eine Ebene mit der ähnlichsten Nachbarfarbe, oder eine ganze Farbe wird mit der ähnlichsten anderen Farbe zusammengelegt. Das gierige Verfahren läuft bis zu 5 Runden, bis das Ergebnis höchstens 3 Ebenen vom Ziel abweicht.
- **Schneller Modus „eine Ebene“** (`layers.py`): Standardmäßig wird jede Farbe einmal über das ganze Bild nachgezeichnet, ohne Arbeit pro einzelner Fläche. Die Farben werden von der größten zur kleinsten gezeichnet, und jede reicht 4 px unter die folgenden, damit keine Lücken entstehen. Das ist auch für Fotos schnell (9–13 s statt Minuten) und braucht wenig Speicher.
- **Formerkennung** (`shapes.py`): Aus den Bildmomenten einer Ebene ergeben sich über die Eigenwerte der Kovarianzmatrix Mittelpunkt, Achsen und Drehung einer Ellipse; für Rechtecke wird das kleinste gedrehte umschließende Rechteck verwendet. Deckt sich die Form mit dem sichtbaren Teil der Ebene zu mindestens 95 % (Intersection over Union), wird sie als echtes `<ellipse>`, `<circle>` oder `<rect>` gespeichert. Weil nur der sichtbare Teil zählt, wird auch eine halb verdeckte Form erkannt und vollständig unter die verdeckende gelegt.
- **Kurvenanpassung** (`curves.py`): Die Pixelkante wird gleichmäßig abgetastet, und an jedem Punkt wird gemessen, wie stark sie abbiegt; Punkte mit großem Winkel sind echte Ecken und bleiben scharf. Zwischen den Ecken wird adaptiv geglättet: sanfte Bögen stark, spitze Stellen wie Krallen kaum. Danach passt der Schneider-Algorithmus (kleinste Quadrate und Newton-Raphson) möglichst wenige kubische Bézier-Kurven an und teilt am Punkt mit dem größten Fehler, solange die Toleranz des Detail-Reglers überschritten ist.
- **Konturlinien** (`strokes.py`): Dunkle Flächen, die lang und dünn sind, werden als Umrisslinien erkannt und mit dem Zhang-Suen-Verfahren auf eine 1 px breite Mittellinie reduziert. Die Distanztransformation misst an jedem Punkt der Mittellinie die Dicke der Linie im Original. Gezeichnet wird ein glattes Band mit dieser gemessenen Dicke um die geglättete Mittellinie, ganz oben über allen Ebenen.
- **Texterkennung** (`text.py`): Buchstaben sind kleine einfarbige Flächen mit dünnen Strichen auf einem einheitlichen Hintergrund, die mit ähnlich großen Nachbarn eine Zeile bilden; jeder Buchstabe wird entlang seiner Zeile aufgerichtet, deshalb funktioniert auch gebogener Text. Gleiche Buchstaben, auch in anderer Schriftgröße, werden durch Drehen und Skalieren übereinandergelegt (Überlappung mindestens 82 %) und zu einer sauberen gemeinsamen Form gemittelt. Jede Kopie wird aus dieser einen Form gezeichnet, und eine Zeile zählt nur als Text, wenn mindestens einer ihrer Buchstaben ein Gegenstück hat.
- **KI-Upscaling** (`upscale.py`): Das neuronale Netz Real-ESRGAN animevideo v3 vergrößert Bilder bis 1024 px auf das Vierfache und macht dabei Kanten scharf und Farben flach. Es läuft als 2,5 MB großes ONNX-Modell auf der CPU, ohne PyTorch und ohne Grafikkarte. Danach wird nicht mehr geschärft: Die Palette kommt aus einer geschärften Kopie, und jedes weiche Kantenpixel geht an die nächste echte Farbe, so bleiben Buchstaben gerade.
- **Qualitätsmessung** (`quality.py`, `gallery.py`): Das fertige SVG wird wieder in Pixel gerendert und Pixel für Pixel mit dem Original verglichen. Gemessen wird der mittlere Farbabstand ΔE im Lab-Raum, Werte unter etwa 2 sind mit bloßem Auge kaum sichtbar. Bei entferntem Hintergrund zählt in der Galerie nur das Objekt, damit die transparente Fläche nicht als Fehler gilt.
- **Preset-Erkennung** (`presets.py`): Das Programm verkleinert das Bild auf 256 px und misst für jedes Pixel, wie stark sich die Helligkeit ändert. Wenn weniger als 30 % der Fläche glatt und einfarbig sind, gilt das Bild als Foto, weil Fotos fast überall Textur und Rauschen haben. Sonst ist es eine Grafik: mit höchstens 12 Hauptfarben ein Logo, mit mehr Farben eine Zeichnung im Cartoon-Stil.

---

## 3. Vorverarbeitung (`preprocess.py`)

### 3.1 Hintergrund freistellen
- **Problem:** Anfangs war der Hintergrund einfach eine weitere Farbe. Weiße Stellen *im* Objekt (z. B. die weißen Flecken an den Händen des Pandas) wurden genauso behandelt wie der weiße Hintergrund und verschwanden.
- **Lösung:**
  1. Hat das Bild einen echten Alphakanal, wird dieser verwendet.
  2. Sonst werden die **Randfarben** ermittelt: k-means mit bis zu 3 Clustern über alle Randpixel. Gezählt werden Cluster mit mehr als 10 % Anteil, dadurch wird auch ein **gemaltes Schachbrettmuster** („falsche Transparenz“) erkannt.
  3. Alle Pixel, die weniger als ΔE 12 von einer Randfarbe entfernt sind, sind Kandidaten.
  4. **Zusammenhangskomponenten** (4er-Nachbarschaft): Nur Kandidatenflächen, die den **Bildrand berühren**, werden Hintergrund.
- **Pointe:** Gleichfarbige Flächen *innerhalb* des Objekts sind nicht mit dem Rand verbunden und bleiben deshalb erhalten.

### 3.2 Arbeitsgröße
- Jedes Bild wird auf etwa **1024 px** gebracht: große Bilder werden verkleinert, kleine **vergrößert** (höchstens 4×).
- **Warum vergrößern?** Dünne Linien (Krallen, Handmarkierungen des Pandas) wären sonst beim Glätten verschwunden.
- Nach dem Vergrößern wird ein Gauß-Weichzeichner angewendet (σ = 0,6 · Faktor), der die Pixeltreppen entfernt. Das spätere Schärfen macht daraus eine glatte Kante.
- Das SVG behält trotzdem die **Originalgröße** (über die `viewBox`).

### 3.3 Rauschen und Texturen entfernen
- **Mean-Shift-Filter** (`cv2.pyrMeanShiftFiltering`, Radius 8, Farbbereich 18): Er glättet Texturen und JPEG-Artefakte, Kanten bleiben dabei scharf.

### 3.4 Kanten schärfen: Toggle-Filter (eigene Implementierung)
- **Problem:** Weiche, kantengeglättete Kanten (Anti-Aliasing) werden sonst zu eigenen dünnen Bändern aus Zwischenfarben.
- **Lösung:** Jedes Pixel übernimmt die **dunkelste oder hellste Farbe** seiner 3×3-Nachbarschaft, je nachdem, welche näher an seiner Helligkeit liegt. Ein weicher Übergang wird so zu einer scharfen Stufe in der Mitte.
- Flache Bereiche (Helligkeitsspanne < 8) bleiben unverändert, sanfte Verläufe werden nicht zerstört.
- Eine dünne Linie behält ihren Kern, weil der Kern ohnehin schon die dunkelste Farbe in der Umgebung ist.
- **Gelernt:** Mehr Durchgänge machten die Nase und die Pupillen des Pandas zackig, deshalb werden heute nur noch 2 Durchgänge verwendet.

### 3.5 Farbpalette: k-means im Lab-Farbraum
- **Lab statt RGB:** Abstände im Lab-Raum (ΔE) entsprechen dem menschlichen Farbempfinden.
- **k-means++** auf einer Stichprobe von 20 000 Objektpixeln, damit es auch bei großen Bildern schnell bleibt.
- **Ähnliche Farben zusammenführen** (agglomerativ): Immer das nächstgelegene Farbpaar wird nach Pixelanzahl gewichtet gemittelt, bis alle Farben mindestens den eingestellten ΔE-Abstand haben (Regler „Color merge range“).
- **Determinismus:** Ein fester Seed (`cv2.setRNGSeed`) sorgt dafür, dass dasselbe Bild immer dieselben Farben und Ebenen ergibt. Vorher lieferte der Panda mal 17, mal 19 Ebenen. Ein anderer Seed erzeugt bewusst eine **Variante** (Button „Regenerate layers“, CLI `--seed`).

### 3.6 Pixel zuordnen und säubern
- Das Bild wird **2× hochskaliert**, und zwar **linear statt kubisch**, damit vergrößerte Kanten nie in dunklere oder hellere Farben „überschwingen“.
- Jedes Pixel bekommt die nächstgelegene Palettenfarbe. **Optimierung:** |x − c|² wird über −2·x·c + |c|² mit `cv2.transform` berechnet, was deutlich schneller ist als NumPy.
- **Mehrheitsfilter (5×5):** Jedes Pixel nimmt die häufigste Farbe seiner Umgebung an. Das entfernt Treppenstufen und Sprenkel. Umgesetzt als Box-Blur pro Farbmaske, also ohne Schleife über einzelne Pixel.

---

## 4. Ebenen statt ausgeschnittener Puzzleteile (`layers.py`)

### 4.1 Das Grundproblem
- Klassische Tracer (auch vtracer) zerlegen das Bild in **Puzzleteile**: Ein Gesicht ist dann eine Fläche mit augenförmigen **Löchern**.
- Folgen: Jede Kante wird doppelt gezeichnet (vom Gesicht und vom Auge), beim Glätten entstehen **Lücken** und Blitzer, und der Nutzer kann die Teile nicht sinnvoll bearbeiten.
- **Unsere Lösung:** Ebenen **wie in Photoshop**, von der größten Fläche unten bis zur kleinsten oben. Darunterliegende Ebenen werden **nicht ausgeschnitten**: Ihre Löcher und Einkerbungen unter den oberen Ebenen werden aufgefüllt, weil sie ohnehin niemand sieht. Das Gesicht bleibt eine ganze Form, die Augen liegen obendrauf.

### 4.2 Aufbau einer Ebene (`build`)
- Jede zusammenhängende Fläche einer Farbe wird zu einer Ebene, sortiert nach Fläche.
- Für jede Ebene gibt es drei Masken:
  - `region`: ihre eigenen Pixel
  - `free`: Pixel, die *später* (darüber) gezeichnet werden, dort darf die Ebene alles bedecken
  - `shape` = `region` ∪ ((Löcher füllen(morphologisches Schließen(region)) ∪ 4 px Überlappung) ∩ `free`)
- **Morphologisches Schließen** füllt Einkerbungen bis zu einem Radius von 0,15 · √Fläche (höchstens 40 px). Aus Geschwindigkeitsgründen läuft es auf einer **verkleinerten Kopie**.
- **4 px Überlappung:** Jede Ebene reicht 4 px unter die Ebenen darüber, damit nach dem Glätten keine Lücken aufgehen.

### 4.3 Säubern vor dem Stapeln (`clean`)
- **Sprenkel entfernen:** Kleine Flächen gehen an die **ähnlichste angrenzende Farbe**, nicht einfach an das nächste Pixel. So schließt sich ein Stück einer blassen Linie dem Rest der Linie an statt dem Hintergrund (die feinen Linien auf dem Pilzhut bleiben durchgehend).
- **Anti-Aliasing-Säume erkennen** (`merge_fringes`): Ein Saum ist eine Fläche, die
  - dünn ist (< 12 px) und klein (< 20 000 px),
  - zwischen **zwei** Flächen liegt, die mindestens 5× größer sind,
  - und deren Farbe eine **Mischung** dieser beiden ist. Geprüft wird das geometrisch: Die Farbe wird auf die Verbindungsstrecke A–B im Lab-Raum projiziert, dabei muss 0,1 < t < 0,9 gelten und der Abstand zur Strecke ≤ ΔE 10 sein.
  - Jedes Saumpixel geht an die Seite, der es farblich näher ist. Die Kante liegt dadurch genau in der Mitte.
  - Ein Stück einer echten dünnen Linie ist kein Saum, weil seine Nachbarn nicht beide viel größer sind.
- **Fotos** zerfallen in Tausende Flächen. Die Mindestgröße wird deshalb verdoppelt, bis höchstens etwa 1 500 Flächen übrig sind.

### 4.4 Gleichfarbige Ebenen zusammenfassen (`group`)
- Eine Form schließt sich einer früheren Ebene derselben Farbe an, wenn **keine dazwischen gezeichnete Ebene sie überlappt**. Das Bild ändert sich dadurch nicht.
- **Ergebnis:** Pilz von 145 Formen auf **18 Ebenen** (alle feinen Hutlinien einer Farbe bilden eine Ebene), Panda von 52 Formen auf **17 Ebenen**.

### 4.5 Ebenenanzahl per Regler (`simplify`)
- Der Nutzer gibt eine Zielanzahl vor. Es werden die **unwichtigsten** Ebenen entfernt, die Kosten berechnen sich als **Fläche × Farbdifferenz (ΔE)**.
- Es gibt zwei Arten von Schritten:
  1. Eine Ebene bekommt die ähnlichste angrenzende Farbe (sie „verschmilzt“ mit ihrer Umgebung).
  2. Eine **ganze Farbe** wird mit der ähnlichsten anderen Farbe zusammengelegt. Nur so lassen sich weniger Ebenen als Farben erreichen.
- **Gieriges Verfahren:** Die billigsten Schritte *pro entfernter Ebene* werden zuerst gewählt. Die Farbe mit der größten Fläche bleibt immer erhalten.
- **Iterativ, bis zu 5 Runden:** Weil eine umgefärbte Form nicht immer einer vorhandenen Ebene beitritt, wird gemessen, wie viel die letzte Runde wirklich gebracht hat („effect“), und entsprechend mehr entfernt. Toleranz: ±3 Ebenen.
- **Schneller gemacht:** Statt alle Ebenen zu bauen und dann die meisten wieder zu löschen, wird das Bild vorher günstig reduziert (etwa 0,7 Farben und 10 Flächen pro gewünschter Ebene).
- **Getestet:** Ziel 12 ergibt Panda 15 / Pilz 12, Ziel 8 ergibt 11 / 10. Die Übereinstimmung bleibt bis etwa 8 Ebenen nah am Original (Panda ΔE 1,39, Pilz ΔE 1,09).

### 4.6 Schneller Modus „eine Ebene“ (`flat`)
- **Problem:** Ein Foto im Ebenenmodus brachte den Server auf **4,7 GB** Speicher, der Pilz brauchte in der Galerie **135 s**.
- **Lösung:** Standardmäßig wird jede Farbe **einmal über das ganze Bild** nachgezeichnet, ohne Arbeit pro Fläche. Die Farben werden von der größten zur kleinsten gezeichnet, jede reicht 4 px unter die folgenden. Dauer: Zeichnungen 2–5 s, Fotos 9–13 s.
- Ebenen gibt es auf Wunsch über den Button **„Separate into layers“**.
- **Schutz:** Bilder mit mehr als 1 000 Flächen (z. B. unruhige Fotos) werden **vor** jedem langsamen Schritt erkannt und bleiben eine Ebene. Die Seite zeigt dazu einen Hinweis statt eines Fehlers.

---

## 5. Erkennung einfacher Formen (`shapes.py`)

- **Problem:** Eine nachgezeichnete Ellipse ist ein Pfad, dessen Umriss leicht wackelt.
- **Lösung:** Ebenen, die fast exakt eine Ellipse, ein Kreis oder ein Rechteck sind, werden zu echtem `<ellipse>`, `<circle>` oder `<rect>`. Das ist perfekt glatt und braucht keinen einzigen Bézier-Punkt.
- **Wie wird die Form bestimmt?**
  - **Ellipse über Bildmomente:** Mittelpunkt aus den Momenten 1. Ordnung, Achsen und Drehung aus den **Eigenwerten und Eigenvektoren der Kovarianzmatrix** (zentrale Momente 2. Ordnung), mit Halbachse = 2·√λ.
  - **Rechteck:** `cv2.minAreaRect` (kleinstes umschließendes gedrehtes Rechteck).
  - **Kreis:** wenn das Achsenverhältnis ≥ 0,97 ist.
- **Akzeptanzkriterium:** Die Überlappung (**Intersection over Union**) mit der Ebene muss ≥ 95 % sein. Ebenen mit Löchern werden übersprungen.
- **Teilweise verdeckte Formen:** Geprüft wird nur der **sichtbare** Teil. Liegt eine Wange halb hinter einem Auge, wird die Ellipse mit `cv2.fitEllipse` nur an den sichtbaren Kantenpunkten angepasst und als vollständige Form darunter gezeichnet.
- **Sonderregel beim Stapeln:** Eigentlich gilt groß unten, klein oben. Wird eine Ebene aber erst dann zur einfachen Form, wenn eine benachbarte Form auf ihr liegen darf, rutscht sie **direkt unter diese Form**. Deshalb liegen die Wangen des Pilzes unter den Augen, obwohl sie kleiner sind.
- **Ergebnis:** Beim Pilz werden beide Augen und beide Wangen zu Ellipsen (ΔE 1,12 → 1,06). Beim Panda werden Körper, Augenflecken, Wangen und Innenohren zu Ellipsen.
- **Ehrliche Grenze:** Exakte Ellipsen weichen etwas von handgezeichneten Ovalen ab. Der Panda ist mit Formen leicht schlechter (1,34 statt 1,16), deshalb lässt sich die Erkennung abschalten.

---

## 6. Kurvenanpassung (`curves.py`), das Herzstück

### 6.1 Problem
- vtracer erzeugte **sehr viele winzige Bézier-Segmente** (teils nur ~2 px lang), selbst an runden Kanten. Beim Pilz waren es rund **8 400 Kurven**.

### 6.2 Algorithmus Schritt für Schritt
1. **Gleichmäßig neu abtasten:** ein Punkt alle 2 px (bei langen Kanten über 800 px alle 3 px, was viel Rechenzeit spart).
2. **Drehwinkel an jedem Punkt** messen: Winkel zwischen den Richtungen 3 Punkte davor und 3 Punkte danach.
3. **Echte Ecken finden:** Punkte mit einem Drehwinkel über der Schwelle (je nach Detailstufe 115° bis 50°), mit Mindestabstand zueinander. Ecken bleiben **scharf**, an ihnen wird die Kontur aufgeteilt.
4. **Glätten zwischen den Ecken:** Gauß-Glättung entlang der Kontur.
   - **Adaptiv:** Ein Gewicht t/(1+t) mit t = Drehwinkel/15° bestimmt, wie viel geglättet wird. Scharfe Stellen (Spitzen, Krallen) werden **weniger** geglättet, sanfte Bögen voll.
5. **Kurven anpassen mit dem Schneider-Algorithmus** (*Graphics Gems*, 1990):
   - Parametrisierung nach Sehnenlänge
   - **Methode der kleinsten Quadrate** für die beiden Kontrollpunkt-Längen bei festen Endpunkten und Tangentenrichtungen
   - **Newton-Raphson**-Schritt, der die Parameter der Punkte verbessert
   - Ist der Fehler zu groß, wird **am Punkt mit dem größten Fehler geteilt** und rekursiv weitergemacht
   - Absicherung: Würden die Griffe länger als die Kontur selbst (das passiert an dünnen Splittern), wird auf Sehne/3 zurückgefallen
6. **Fehlertoleranz pro Punkt:** an scharfen Stellen enger, und höchstens 4 % der Formgröße, damit kleine Formen ihre Details behalten.
7. **Kleine Konturen** (< 120 Punkte) bekommen einen schnellen **Catmull-Rom-Spline** statt der vollen Anpassung. Das ist um ein Vielfaches schneller, und dort gibt es ohnehin keine Ecken, die man erhalten müsste.

### 6.3 Ein Regler für alles: „Curve detail“ (1–10)
- Die Toleranz sinkt **geometrisch** von 6 px (Stufe 1) auf 0,6 px (Stufe 10). Gleichzeitig ändern sich die Glättung (σ 4 → 0,5) und der Eckwinkel.
- Glättung und Toleranz werden **mit der Bildgröße skaliert** (Fix in PR #4). Sonst wurden kleine Details wie Krallen weggerundet.
- **Ergebnis Pilz:** ~8 400 Kurven (vtracer) → **940** bei Stufe 1, **1 130** bei Stufe 4, 4 800 bei Stufe 10.

### 6.4 Parallelisierung
- Formen sind voneinander unabhängig, deshalb läuft die Anpassung mit einem `ProcessPoolExecutor` **auf allen CPU-Kernen**.
- **Größte Formen zuerst**, damit am Ende kein Prozess mit einer großen Form allein übrig bleibt.
- Eine Farbe eines Fotos kann Tausende Kanten haben, sie wird deshalb in Jobs zu je 150 Kanten aufgeteilt.
- **Ein Mathe-Thread pro Prozess** (`OMP_NUM_THREADS=1` usw.), weil die Prozesse sonst um die Kerne konkurrieren.

---

## 7. Konturlinien als glatte Bänder (`strokes.py`)

### 7.1 Problem
- Eine nachgezeichnete Umrisslinie ist eine gefüllte Fläche, deren **zwei Kanten getrennt geglättet** werden. Die Dicke wackelt dadurch.

### 7.2 Lösung
1. **Kandidaten:** dunkle Farben (Lab-Helligkeit < 50), die **lang und dünn** sind: Breite zwischen 4 px und 1,5 % der Bildgröße, Länge ≥ 6× Breite und ≥ 3 % der Bildgröße, Fläche ≤ 1,8 × Breite × Länge (ein massiver Fleck hätte viel mehr Fläche).
2. **Skelettierung mit dem Zhang-Suen-Verfahren** (selbst vektorisiert in NumPy) reduziert die Form auf eine 1 px breite Mittellinie. Zur Beschleunigung läuft sie auf halber Auflösung (4× schneller).
3. **Dicke messen:** Die **Distanztransformation** liefert an jedem Mittellinienpunkt die halbe Breite, die Dicke wird also aus dem Original gemessen.
4. **Skelett als Graph:** Knoten sind Endpunkte und Kreuzungen, die Linien dazwischen werden abgelaufen. Ein **diagonaler Nachbar zählt nicht**, wenn ein gerader Nachbar die beiden schon verbindet. Sonst sähe jede Treppenstufe wie eine Kreuzung aus. Kurze Seitenäste an Ecken werden verworfen.
5. **Dicke Teile bleiben Flächen:** Stellen, die dicker als 2,5× die Linienbreite sind (z. B. die Nase des Pandas am Mund), werden vom Band ausgenommen.
6. **Band zeichnen:** Die Mittellinie wird geglättet, die Dicke **noch stärker** (Dicke ändert sich langsamer als die Richtung). Entlang der Normalen wird ±Dicke/2 versetzt, die Enden bekommen **runde Kappen**. Geschlossene Linien werden als zwei Schleifen in entgegengesetzter Richtung ausgegeben, damit die Mitte offen bleibt.
7. An Kreuzungen wird die Dicke auf 2× die normale Breite begrenzt, damit die Kreuzung nicht anschwillt.
8. Die Linien werden **ganz oben** gezeichnet, die Ebenen darunter reichen unter sie. Kleine Reste von Anti-Aliasing direkt neben der Linie werden ausgeblendet.
- **Reihenfolge dunkelste Farbe zuerst:** So wird der hellere Saum einer Linie nicht als zweite Linie daneben erkannt.

### 7.3 Was nicht funktioniert hat
- **Versuch:** alle Anti-Aliasing-Pixel der dunkleren Farbe zuschlagen.
- **Ergebnis:** Alle dunklen Flächen wuchsen, der Panda verschlechterte sich von ΔE 1,15 auf **1,83**. Der Versuch wurde zurückgenommen. Stattdessen wird die Breite der Kontur gemessen (siehe oben).
- **Ergebnis der neuen Lösung:** Die Pilzkontur ist glatt und an denselben Stellen dicker oder dünner wie im Original, die Übereinstimmung bleibt gleich (ΔE 0,94–1,01). Der Mund des Pandas ist eine saubere Linie, die zu den Enden hin dünner wird.

---

## 8. Optionales KI-Upscaling (`upscale.py`)

- **Problem:** Kleine oder unscharfe Zeichnungen und Logos haben zu wenige Pixel. Dünne Konturen, Buchstaben und Krallen gehen beim Vektorisieren verloren.
- **Lösung:** Das neuronale Netz **Real-ESRGAN animevideo v3** (SRVGGNetCompact, 0,6 Mio. Parameter, 4× Vergrößerung) macht Kanten scharf und Farben flach.
- **Engineering-Entscheidungen:**
  - Die Gewichte (`.pth`) wurden mit PyTorch einmalig nach **ONNX** konvertiert (Opset 17, dynamische Bildgröße). Das Projekt braucht dadurch **kein PyTorch** und keine Grafikkarte, sondern nur `onnxruntime` auf der CPU. Das Modell ist nur 2,5 MB groß.
  - Das Netz sieht nur die Farben. Der Alphakanal wird separat klassisch skaliert.
  - Die Funktion ist **optional**: Fehlt `onnxruntime`, läuft alles andere normal weiter.
  - Angeboten wird sie nur bis 1024 px, größere Bilder haben schon genug Pixel.
- **Problem danach: wellige Buchstaben** (PR #13)
  - Das übliche Kantenschärfen ordnet jedes Kantenpixel einzeln zu, was Buchstaben zackig macht.
  - **Lösung:** Nach dem KI-Upscaling wird nicht geschärft. Die **Palette** wird aus einer geschärften Kopie bestimmt (damit Zwischenfarben keine eigenen Farben werden), die **Zuordnung** läuft aber auf dem weichen Bild. Jedes Kantenpixel geht an die nächste echte Farbe, die Kante liegt genau in der Mitte des weichen Übergangs, und **gerade Kanten bleiben gerade** (z. B. der Text auf `img/umbrella.png`).
- **Messung** (Zeichnungen auf 256 px verkleinert und als JPEG gespeichert, Fehler gegen das Original in voller Größe):

| Bild | ohne KI | mit KI |
|---|---|---|
| Logo | 14,5 | **9,2** |
| Dino | 15,9 | **10,4** |
| Pixel-Art-Figur | 8,6 | **6,0** |

- **Laufzeit (Apple Silicon, CPU):** deutlich unter 1 s bei 256 px, ~3 s bei 880 × 660 px, ~5 s bei 1024² px.

---

## 9. Qualität messbar machen (`quality.py`, `gallery.py`)

- **Idee:** Nicht nach Gefühl bewerten. Das SVG wird **zurück in Pixel gerendert** und pixelweise mit dem Original verglichen.
- **Eigener Mini-Renderer** in `quality.py`: Pfade, Ellipsen, Kreise und Rechtecke werden geparst und mit OpenCV gefüllt, mit Subpixel-Genauigkeit (`shift=4`, also 1/16 Pixel). Kennzahl: **mittleres ΔE im Lab-Raum** (unter ~2 ist kaum sichtbar).
- **Galerie:** Hier wird mit `resvg` (einem richtigen SVG-Renderer) gerendert, der Fehler liegt im Bereich 0–255.
  - **Fairness-Problem:** Bei entferntem Hintergrund würde der transparente Bereich mit der alten Hintergrundfarbe verglichen und als Fehler zählen. **Lösung:** Gemessen wird nur auf dem **Objekt**, also den als Objekt behaltenen Pixeln plus allem, was das SVG malt.
  - Bewertung: **good** < 5, **okay** < 12, sonst **poor**.
- Diese Messung war das **Werkzeug für jede Entscheidung** im Projekt: Jede Änderung wurde mit Vorher/Nachher-Zahlen belegt (siehe die ΔE-Werte in diesem Dokument).

---

## 10. Automatische Presets (`presets.py`)

| Preset | Farben | Merge (ΔE) | Detail | Hintergrund | Für |
|---|---|---|---|---|---|
| `logo` | 8 | 12 | 6 | entfernt | Logos, Icons |
| `cartoon` | 16 | 8 | 4 | entfernt | Zeichnungen, Figuren |
| `photo` | 32 | 4 | 5 | behalten | Fotos, Szenen |

- **Erkennung** auf einer 256-px-Version des Bildes:
  1. **Sobel-Gradient:** Sind weniger als 30 % des Bildes flach (Gradient < 8), ist es ein **Foto**. Fotos haben fast überall Textur.
  2. Sonst werden die Farben auf 4 Bit pro Kanal reduziert (damit Rauschen und Anti-Aliasing nicht zählen). Decken ≤ 12 Farben 95 % des Bildes ab, ist es ein **Logo**.
  3. Sonst ist es ein **Cartoon**.

---

## 11. Backend und Performance (`app.py`, `gallery.py`)

### 11.1 Mehrstufiger Cache, damit die Regler „live“ wirken
- **Beim Upload, einmalig:** Vorverarbeitung (Größe, Mean-Shift, Schärfen).
- **Schlüssel (Farben, Merge, Hintergrund, Seed):** Ergebnis der Farbreduktion.
- **Plus (Sprenkel, Konturen, Formen, Ebenen, Trennen):** Ergebnis der Ebenenbildung.
- **Nur der Detail-Regler geändert:** Es wird nur die Kurvenanpassung neu gerechnet (~1 s).
- Höchstens 8 Bilder im Speicher, das älteste fliegt zuerst raus (LRU über `OrderedDict`).

### 11.2 Galerie im Hintergrund, mit Vorrang für den Nutzer
- Beim Serverstart konvertiert **ein Hintergrund-Thread** alle Bilder aus `img/`.
- **Priorität:** Ein Zähler mit `threading.Condition` sorgt dafür, dass die Galerie pausiert, solange eine Anfrage von der Hauptseite läuft, auch zwischen den langsamen Schritten eines Bildes.
- **Versionierung der Kacheln:** Die Fehlermessung läuft asynchron. Ein veraltetes Ergebnis überschreibt nie ein neueres.
- Ergebnisse mit und ohne KI-Upscaling bekommen **getrennte Kacheln** und lassen sich so direkt vergleichen.

### 11.3 Gesammelte Beschleunigungen
- Paralleles Kurvenfitting auf allen Kernen, ein Mathe-Thread pro Prozess
- Morphologisches Schließen auf verkleinerter Kopie
- Skelettierung auf halber Auflösung
- Schneller Spline für kleine Konturen, nur 2 Verfeinerungsdurchgänge im Schneider-Algorithmus
- Bildsäuberung einmal statt in jeder Neuberechnungsrunde
- Vorab-Reduktion der Farben und Flächen bei einer Zielanzahl an Ebenen
- **Ergebnis beim Trennen in 15 Ebenen:** Dino 24 s → **15 s**, OIP 28 s → **10 s**, Pilz 6,5 s

---

## 12. Weboberfläche (`static/`)

- **Drag & Drop** zum Hochladen. Das Preset wird automatisch gewählt und lässt sich per Button wechseln.
- **Vorher/Nachher-Vergleich** mit verschiebbarem Trenngriff (CSS `clip-path: inset(...)`).
- **Klick-Zoom 4×**, bei dem der angeklickte Punkt unter dem Mauszeiger bleibt, dazu Verschieben per Ziehen. Wenige Pixel Bewegung zählen noch als Klick.
- **Ebenen-Panel wie in Photoshop:** oberste Ebene zuerst, Vorschaubild auf die Ebene gezoomt, Name („Ellipses ×2“, „Outline“) und **Auge-Button** zum Ausblenden. Ausgeblendete Ebenen fehlen auch im Download.
  - Technisch ist jede Ebene eine SVG-Gruppe `<g data-layer data-name data-box>`, die Ebenen sind also auch in anderen Programmen (Illustrator, Inkscape) sichtbar.
- **PNG-Export ×1/×2/×4/×8:** Breite und Höhe des SVG werden auf die Zielgröße gesetzt, dann zeichnet der Browser es auf ein Canvas. Ein PNG aus dem Vektor bleibt deshalb auch bei ×8 scharf. Maximal 8192 px pro Seite (Grenze der Browser).
- **Galerie:** drei Ansichten (Kacheln / Diagramme / Tabelle), Vollbild-Viewer mit Pfeiltasten, **„✎ Edit“** öffnet ein Galeriebild mit allen Einstellungen wieder auf der Hauptseite.
- Im echten Browser getestet (Edge, automatisiert mit Playwright).

---

## 13. Entwicklungsverlauf: Problem → Lösung

| Runde | Problem | Lösung | Ergebnis |
|---|---|---|---|
| 1 | Zackige Kanten, riesige Dateien | Mean-Shift, Lab-k-means, ΔE-Merge, 2× + Mehrheitsfilter | Pilz: 726 → 244 Pfade, 862 KB → 247 KB |
| 2 | Tausende Mini-Segmente von vtracer | eigene Kurvenanpassung (Ecken + Schneider) | ~8 400 → 940–1 130 Kurven |
| 3 | Weiß im Objekt = Hintergrund | Freistellen über Randfarben + Zusammenhang mit dem Rand | transparente SVGs, weiße Details bleiben |
| 4 | Dünne Details verschwinden | Arbeitsgröße 1024 px, Toggle-Filter, adaptive Glättung | Pilz ΔE 6,6 → ~0,95, Panda 1,43 → ~1,2 |
| 5 | Wackelige Konturdicke | Skelett + gemessene Dicke + Band | glatte Linien mit Originaldicke |
| 6 | Ovale wackeln | Formerkennung über Momente / IoU | Augen und Wangen als echte Ellipsen |
| 7 | Puzzleteile mit Löchern | **vtracer entfernt**, eigene gestapelte Ebenen | schneller, Ebenen bearbeitbar |
| 8 | Zu viele Ebenen, nicht reproduzierbar | Zusammenfassen, Ebenen-Regler, fester Seed | Pilz 145 → 18 Ebenen |
| 9 | Fotos blockieren den Server (4,7 GB) | Standardmodus „eine Ebene“, Detailgrenze | Fotos 9–13 s statt Minuten |
| 10 | Kleine, unscharfe Bilder | Real-ESRGAN über ONNX | Fehler −30 bis −37 % |
| 11 | Wellige Buchstaben nach dem KI-Upscaling | Palette geschärft, Zuordnung weich | gerade Kanten bei Text |

**Verworfene Ansätze (gut für die Präsentation, sie zeigen den Lernprozess):**
- Anti-Aliasing-Pixel pauschal zur dunklen Farbe schlagen: verschlechterte den Panda von 1,15 auf 1,83.
- Vier feste Presets („Logo / flat“, „Illustration“, …): ersetzt durch automatische Erkennung.
- „Recommended for this image“ (probiert mehrere Einstellungen durch und wählt die einfachste passende): zu langsam (~13 s), wieder entfernt.
- Stärkeres Kantenschärfen: machte weiche Formen zackig, deshalb heute nur noch 2 Durchgänge.

---

## 14. Grenzen und ehrliche Einschätzung

- Die Hintergrunderkennung arbeitet vom Rand aus. Berührt das Objekt den Rand in Hintergrundfarbe, wird dieser Teil mit entfernt (dafür gibt es `--keep-background`).
- Bei sehr wenigen Ebenen (≈ 5) gehen Details verloren, die für Menschen wichtig sind (Augen des Pilzes, Pupillen des Pandas), weil die Auswahl nur Fläche und Farbe betrachtet, nicht die Bedeutung.
- Formen mit Löchern werden nicht als einfache Form erkannt.
- Sehr dünne, spitz zulaufende Linienenden (< 4 px) bleiben gefüllte Flächen.
- Foto-SVGs sind groß (oft ~1 MB), Vektorisierung ist für Fotos grundsätzlich nur begrenzt sinnvoll.
- Das KI-Upscaling verändert bei Fotos und Aquarellen den Stil (dunklere Konturen, flachere Schattierung).
- Die Galerie liegt nur im Speicher und wird bei jedem Start neu aufgebaut.

---

## 15. Vorschlag für den Folienaufbau (ca. 12–15 Folien)

1. **Titel:** Raster → SVG, Team, Ziel
2. **Motivation:** Warum Vektoren? Skalierbarkeit, Dateigröße, Bearbeitbarkeit. Was ist schwer daran?
3. **Live-Demo kurz anreißen** oder Vorher/Nachher-Bild (Pilz oder Panda mit Zoom)
4. **Pipeline-Übersicht** (Diagramm aus Abschnitt 2)
5. **Vorverarbeitung:** Freistellen über den Rand, Lab-Farbraum, k-means, Toggle-Filter
6. **Ebenen statt Puzzleteile:** Gesicht mit Löchern vs. Gesicht mit Augen darauf (anschaulichstes Bild!)
7. **Kurvenanpassung:** Ecken finden, adaptiv glätten, Schneider-Algorithmus; 8 400 → 940 Kurven
8. **Formerkennung:** Momente/Eigenwerte → Ellipse, IoU ≥ 95 %, verdeckte Wange unter dem Auge
9. **Konturlinien:** Zhang-Suen-Skelett + Distanztransformation → Band mit Originaldicke
10. **KI-Upscaling:** Real-ESRGAN, ONNX ohne PyTorch, Messtabelle, Trick gegen wellige Buchstaben
11. **Qualität messen:** SVG zurückrendern, ΔE, Bewertung good/okay/poor
12. **Performance:** Cache-Stufen, Parallelisierung, 4,7 GB → Modus „eine Ebene“
13. **Entwicklungsverlauf:** Tabelle aus Abschnitt 13, inklusive verworfener Ansätze
14. **Grenzen und Ausblick**
15. **Live-Demo**

### Demo-Ablauf (Vorschlag)
1. `uvicorn app:app --reload` starten und http://127.0.0.1:8000 öffnen
2. `img/mushroom.jpg` per Drag & Drop hochladen, das Preset wird automatisch erkannt
3. Trenngriff ziehen und hineinzoomen: glatte Kanten
4. **„Separate into layers“**, dann im Ebenen-Panel eine Ebene ausblenden, den Ebenen-Regler bewegen und „Regenerate layers“ zeigen
5. Detail-Regler bewegen: schnelle Neuberechnung (nur Kurven), Kurvenanzahl ändert sich
6. `img/umbrella.png` mit und ohne **AI upscale**: Text wird gerader
7. **Galerie:** Diagrammansicht, Fehlerbewertung, „✎ Edit“
8. Download als SVG und als PNG ×8 und zeigen, dass das PNG scharf ist

---

## 16. Schlagworte für Rückfragen

Lab-Farbraum und ΔE · k-means++ · agglomeratives Zusammenführen · Mean-Shift-Filter · Toggle-Filter · Mehrheitsfilter · Zusammenhangskomponenten · morphologisches Schließen · Distanztransformation · Zhang-Suen-Skelettierung · Bildmomente und Eigenwerte · Intersection over Union · Schneider-Kurvenanpassung · Methode der kleinsten Quadrate · Newton-Raphson · Catmull-Rom-Spline · gieriger Algorithmus mit Kostenfunktion · ProcessPoolExecutor · LRU-Cache · Real-ESRGAN · ONNX Runtime · FastAPI
