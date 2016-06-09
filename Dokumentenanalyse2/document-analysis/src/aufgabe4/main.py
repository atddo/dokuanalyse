from aufgabe4.pca import PCAExample

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # IGNORE:unused-import
import numpy as np
from features import WordListNormalizer, BagOfWords, TopicFeatureTransform, RelativeTermFrequencies
from evaluation import CrossValidation
from classification import KNNClassifier
import itertools
from corpus import CorpusLoader

def aufgabe4():
    
    #
    # Mit dem Naechster Nachbar Klassifikator wurde ein Dokumente zu einer Klassen zugeordnet,
    # indem zunaechst aehnliche Dokumente aus einer Trainingsdatenbank ermittelt wurden.
    # Ueber die Klassenzugehoerigkeit dieser aehnlichen Dokumente aus dem Training
    # wurde dann das unbekannte Dokument einer Klasse zugeordnet.
    # Dabei wurden aber noch keine Zusammenhaenge zwischen einzelnen Woertern analysiert
    # und beruecksichtigt. Daher geht es nun um Topic Modelle. Topic Modelle beschreiben
    # diese Zusammenhaenge durch einen mathematischen Unterraum. Die Vektoren, die
    # diesen Unterraum aufspannen, sind die Topics, die jeweils fuer typische Wort-
    # konfigurationen stehen. Dokumente werden nun nicht mehr durch Frequenzen von
    # Woertern repraesentiert, sondern als Linearkombination von Topics im Topic 
    # Vektorraum. Es ist zu beachten, dass fuer die Topic-Modellierung keine Informationen
    # ueber die Dokumentenkategorien benoetigt wird.
    #
    # Um ein besseres Verstaendnis fuer diese mathematischen Unterraeume zu entwickeln,
    # schauen wir uns zunaechst die Hauptkomponentenanalyse an.
    #
    
    # Ein 3D Beispieldatensatz wird aus einer Normalverteilung generiert.
    # Diese ist durch einen Mittelwert und eine Kovarianzmatrix definiert
    mean = np.array([10, 10, 10])
    cov = np.array([[3, .2, .9],
                    [.2, 5, .4],
                    [.9, .4, 9]])
    n_samples = 1000
    limits_samples = ((0, 20), (0, 20), (0, 20))
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    # Plotten der Beispieldaten
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    PCAExample.plot_sample_data(samples, ax=ax)
    PCAExample.set_axis_limits(ax, limits=limits_samples)
    
    # In der Klasse PCAExample wird ein Unterraum mittels Hauptkomponentenanalyse
    # statistisch geschaetzt. Der Vektorraum wird beispielhaft visualisiert.
    pca_example = PCAExample(samples, target_dim=3)
    pca_example.plot_subspace(limits=limits_samples, color='r', linewidth=0.05, alpha=0.3)
    plt.show()

    # Nun wird die Dimension des Unterraums reduziert. 
    # Implementieren Sie die Reduktion im Konstruktor von PCAExample. Der neue 
    # Vektorraum wird wieder visualisiert.
    pca_example_2d = PCAExample(samples, target_dim=2)
    pca_example_2d.plot_subspace(limits=limits_samples, color='b', linewidth=0.01, alpha=0.3)
    
    # Transformieren Sie nun die 3D Beispieldaten in den 2D Unterraum.
    # Implementieren Sie dazu die Methode transform_samples. Die Daten werden
    # dann in einem 2D Plot dargestellt.
    #
    # Optional: Verwenden Sie Unterraeume mit Dimensionen 3, 2 und 1. Transformieren
    # und plotten Sie die Daten.
    #
    # Optional: Generieren und transformieren Sie weitere 3D Beispieldaten. Benutzen Sie 
    # dabei auch andere Parameter fuer die Normalverteilung.
    #
    # Optional: Visualisieren Sie die transformierten 2D Daten auch in dem vorherigen
    # 3D Plot.
    samples_2d = pca_example_2d.transform_samples(samples)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    PCAExample.plot_sample_data(samples_2d, ax=ax)
    PCAExample.set_axis_limits(ax, limits=((-10, 10), (-10, 10)))
 
    plt.show()
        
    # Berechnen Sie nun die Kovarianzmatrix der transformierten Daten.
    # Welche Eigenschaften hat diese Matrix? (Dimension, etc.)
    # In welcher Groessenordnung liegen die einzelnen Eintraege? Erklaeren Sie das
    # anhand des vorherigen 2D Plots.
    # Vergleichen Sie das Ergebnis mit der Kovarianzmatrix, die oben zur Generierung
    # der Daten verwendet wurde.
    # Was erwarten Sie fuer den Mittelwert der transformierten Daten (noetig fuer 
    # die Berechnung der Kovarianzmatrix) ?
    #
    # Verwenden Sie bei der Berechnung nicht die eingebaute numpy.cov Funktion
    # (hoechstens zur Kontrolle, achten Sie dabei auf den "bias" Parameter)
    # Verwenden Sie bei der Berechnung keine Schleifen, sondern nur Matrixoperationen.
    # Erklaeren Sie die Vorgehensweise.
    
    zeilen, spalten = samples_2d.shape
    print zeilen, spalten
    mean_xy = np.mean(samples_2d, axis=0)
 
    samples_2d_mf = samples_2d - mean_xy
    kovarianz_mat = np.dot(samples_2d_mf.T, samples_2d_mf)/(zeilen-1)
     
    print "kovarianz ausgerechnet"
    print kovarianz_mat
  
    kovarianz_referenz = np.cov(samples_2d, rowvar=0)
    print "kovarianz referenz"
    print kovarianz_referenz
    
    
    #
    # Latent Semantic Indexing
    #
    # Im folgenden soll ein Topic-Raum mittels Latent Semantic Indexing verwendet
    # werden. Das Prinzip geht unmittelbar auf die Hauptkomponentenanalyse zurueck.
    # Siehe: http://lsa.colorado.edu/papers/JASIS.lsi.90.pdf (Seite 12)
    # Grundsaetzlicher Unterschied ist, dass der Unterraum nicht durch eine Eigenewert-
    # analyse der Kovarianzmatrix bestimmt wird. Stattdessen ergibt sich der Unterraum
    # aus einer Zerlegung der Term-Dokument (!) Matrix mit einer Singulaerwertzerlegung.
    # Man kann zeigen, dass diese Singulaerwertzerlegung implizit einer Eigenwert-
    # analyse einer Termkorrelationsmatrix entspricht. Deren Berechnung unterscheidet 
    # sich von der Berechnung der Kovarianzmatrix insbesondere darin, dass die Daten 
    # nicht vom Mittelwert befreit werden. 
    # Sei t die Anzahl der Terms (Groesse des Vokabulars), d die Anzahl der Dokumente,
    # m der Rang von X (Maximale Anzahl von Topics, die sich aus X bestimmen lassen).
    # D' ist die Transponierte von D.
    # 
    #   X    =    T    *    S    *    D'
    # t x d     t x m     m x m     m x d
    #
    # In Analogie zur Hauptkomponentenanalyse findet man nun die Vektoren, die
    # den Unterraum aufspannen, in den Spalten von T. Die Matrix S hat nur Eintraege
    # auf der Diagonalen und enthaelt die Singulaerwerte zu den Spaltenvektoren in
    # T. (T und D enthalten die linken respektive rechten Singulaervektoren.) 
    # Die Singulaerwerte entsprechen den Eigenwerten in der Hauptkomponentenanalyse.
    # Sie sind ein Mass fuer die Variabilitaet in den einzelnen Topics. Bei D handelt
    # es sich um die Koeffizienten der d Dokumente im Topic Raum (Ergebnis der 
    # Transformation von den Bag-of-Words Repraesentationen aus X in den Topic Raum.)
    #
    # 
    # Aus der Singulaerwertzerlegung (Formel oben) ergibt sich, wie man einen Topic-
    # Raum statistisch aus Beispieldaten schaetzt. Um den Topic-Raum aber mit unbekannten Daten
    # zu verwenden, muessen diese in den Topic-Raum transformiert werden. 
    # Stellen Sie dazu die obige Formel nach D um. Die zu transformierenden Bag-of-Words
    # Repaesentationen koennen dann fuer X eingesetzt werden. Dabei ist wichtig zu
    # beachten:
    # Die Spaltenvektoren in T sind orthonormal (zueinander) T' * T = I
    # Die Spaltenvektoren in D sind orthonormal (zueinander) D' * D = I
    # Dabei ist I die Einheitsmatrix, T' und D' sind die Transponierten in T und D.
    # Fuer Matrizen A und B gilt: (A * B)' = B' * A'
    #
    # Ueberlegen Sie wie die Transponierte einer Matrix gebildet wird und was das
    # fuer eine Matrix bedeutet, deren Eintraege nur auf der Hauptdiagonalen von
    # 0 verschieden sind.
    #
    # Erlaeutern Sie die Funktion der einzelnen Matrizen in der sich ergebenden
    # Transformationsvorschrift. 
    #
    
      
    # Das Schaetzen eines Topic-Raums soll an folgendem einfachen Beispiel veranschaulicht
    # werden. Sei dazu bow_train eine Dokument-Term Matrix mit 9 Dokumenten und 3 Terms.
    # Welcher Zusammenhang zwischen den Terms faellt Ihnen auf? 
    bow_train = np.array([[2, 5, 0],
                          [4, 1, 0],
                          [3, 3, 1],
                          [9, 8, 2],
                          [1, 5, 3],
                          [0, 7, 9],
                          [2, 9, 6],
                          [0, 2, 3],
                          [5, 3, 3]])
    
    # Zerlegung der Dokument-Term Matrix mit der Singulaerwertzerlegung
    T, S_arr, D_ = np.linalg.svd(bow_train.T, full_matrices=False)
    S = np.diag(S_arr)
    print 'Matrix T, Spaltenvektoren definieren Topic Raum' 
    print T
    print 'Matrix S, Singulaerwerte zu den Vektoren des Topic Raums' 
    print S
    print 'Matrix D, Koeffizienten der Termvektoren in bof im Topic Raum'
    print D_.T
    
    # Transformieren Sie nun die folgenden Termvektoren in den Topic Raum
    # Was erwarten Sie fuer die Topic Zugehoerigkeiten?
    
    bow_test = np.array([[5, 0, 0],
                     [0, 5, 0],
                     [0, 0, 5],
                     [5, 5, 0],
                     [0, 5, 5]])
    
    n_topics = 3
    top_feature_trans = TopicFeatureTransform(n_topics)
    top_feature_trans.estimate(bow_train, bow_train)
    bow_transformed = top_feature_trans.transform(bow_test)
     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    PCAExample.plot_sample_data(bow_train, annotations=bow_train, ax=ax)
    PCAExample.set_axis_limits(ax, limits=((0, 10), (0, 10), (0, 10)))
     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    PCAExample.plot_sample_data(bow_transformed, annotations=bow_test, ax=ax)
    PCAExample.set_axis_limits(ax, limits=((-1, 1), (-1, 1), (-1, 1)))
    plt.show()
        
    #
    # Warum lassen sich die Koeffizienten der Termvektoren so schwer interpretieren?
    # 
    # Um eine bessere Vorstellung von der Bedeutung der einzelnen Topics zu bekommen,
    # plotten Sie die Bag-of-Words Repraesentationen sowie die Topic-Koeffizienten der 
    # Trainingsdaten (bow_train) und der Testdaten (bow_test) in verschiedenen Farben.
    # Erstellen Sie dazu jeweils einen Plot fuer Bag-of-Words Repraesentationen und einen
    # Plot fuer Topic-Koeffizienten. Achten Sie auf eine geeignete Skalierung der Axen.
    # Um die Datenpunkte in den beiden Plots besser einander zuordnen zu koennen, plotten
    # Sie zusaetzlich die Termfrequenzen neben jeden Datenpunkt (als Annotation).  
    # Mehrere Daten (Trainings-, Testdaten, Annotationen) lassen sich in einem gemeinsamen 
    # Plot darzustellen indem Sie die Funktion 'hold' des Axes Objekts mit dem Parameter 
    # 'True' aufrufen. Zum Erstellen der Plots orientieren Sie sich an den entsprechenden 
    # Funktionen aus dem Beispiel zur Hauptkomponentenanalyse (oben). Schauen Sie sich 
    # auch deren weitere Parameter (und zusaetzlich vorhandene Hilfsfunktionen) an. 
    
    #Bag of words plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    PCAExample.plot_sample_data(bow_train, color='r', annotations=bow_train, ax=ax)
    fig.hold(True)
    PCAExample.plot_sample_data(bow_test, color='g', annotations=bow_test, ax=ax)
    PCAExample.set_axis_limits(ax, limits=((0, 10), (0, 10), (0,10)))
    fig.hold()
     
    #koeffizienten plot
    T_test, S_arr_test, D_test_ = np.linalg.svd(bow_test.T, full_matrices=False)
    S_test = np.diag(S_arr_test)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    PCAExample.plot_sample_data(D_.T, color='r', annotations=bow_train, ax=ax)
    fig.hold(True)
    PCAExample.plot_sample_data(D_test_.T, color='g', annotations=bow_test, ax=ax)
    PCAExample.set_axis_limits(ax, limits=((-1, 1), (-1, 1), (-1,1)))
    fig.hold()
    
        
    #
    # Fuehren Sie nun eine Dimensionsreduktion der Trainings und Testdaten auf zwei 
    # Dimensionen durch und plotten Sie die Topic-Koeffizienten (inkl. Bag-of-Words 
    # Annotationen). Vergleichen Sie alle drei Plots miteinander. Welchen Effekt hat 
    # die Topic Modellierung im Bezug auf typische Termkonfigurationen?
    #
    # Optional: Transformieren Sie die Daten in einen Topic-Raum mit Dimension Eins
    # und plotten Sie die Koeffizienten inkl. deren Bag-of-Words Annotationen. 
    #
    
    n_topics = 2
    top_feature_trans = TopicFeatureTransform(n_topics)
    top_feature_trans.estimate(bow_train, None)
    
    S_inv = np.linalg.inv(S)
    S_inv_test = np.linalg.inv(S_test)
    
    T = T[:n_topics]
    S_inv = S_inv[:n_topics,:n_topics]
    T_test = T_test[:n_topics]
    S_inv_test = S_inv_test[:n_topics,:n_topics]
    
    data_trans = np.dot(np.dot(bow_train, T.T), S_inv)
    data_trans_test = np.dot(np.dot(bow_test, T_test.T), S_inv_test)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    PCAExample.plot_sample_data(data_trans, color='r', annotations=bow_train, ax=ax)
    fig.hold(True)
    PCAExample.plot_sample_data(data_trans_test, color='g', annotations=bow_test, ax=ax)
    PCAExample.set_axis_limits(ax, limits=((-1, 1), (-1, 1)))
    fig.hold()
    plt.show()
    
    #
    # Integrieren Sie nun die Topic-Raum Modellierung mittels Singulaerwertzerlegung 
    # in die Kreuzvalidierung auf dem Brown Corpus. Berechnen Sie dabei fuer
    # jede Aufteilung von Training und Test einen neuen Topic-Raum. Transformieren
    # Sie die Bag-of-Words Repraesentationen und fuehren Sie die Klassifikation
    # wie zuvor mit dem Naechster-Nachbar-Klassifikator durch. Verwenden Sie dabei
    # verschiedene Distanzmasse und evaluieren Sie die Klassifikationsfehlerrate
    # fuer verschiedene Dimensionalitaeten des Topic-Raums. Die anderen Parameter
    # waehlen Sie gemaess der besten bisherigen Ergebnisse.
    #
    # Implementieren Sie die Klasse TopicFeatureTransform im features Modul
    # und verwenden Sie sie mit der CrossValidation Klasse (evaluation Modul).
    #
    # Optional: Fuehren Sie eine automatische Gridsuche ueber den kompletten Paramterraum
    # durch. Legen Sie sinnvolle Wertebereiche und Schrittweiten fuer die einzelnen
    # Parameter fest. Wie lassen sich diese bestimmen?
    #
    # Optional: Passen Sie das Suchgrid dynamisch gemaess der Ergebnisse in den einzelnen
    # Wertebereichen an.

    CorpusLoader.load()
    brown = CorpusLoader.brown_corpus()
    
    normalized_words = WordListNormalizer().normalize_words(brown.words())[1]
    
    vocab_size = 2000
    distance_function="cityblock"
    knn=6
    vocabulary = BagOfWords.most_freq_words(normalized_words, vocab_size)
    word_bag = BagOfWords(vocabulary, RelativeTermFrequencies())

    bow_mat = {}
    for cat in brown.categories():
        bow_mat[cat] = [(brown.words(fileids=doc)) for doc in brown.fileids(categories=cat)]
    category_dic = word_bag.category_bow_dict(bow_mat)

    cross_validator = CrossValidation(category_dic, 5)

    classificator = KNNClassifier(knn, distance_function)
    print cross_validator.validate(classificator, TopicFeatureTransform(vocab_size))
    
    
if __name__ == '__main__':
    aufgabe4()
