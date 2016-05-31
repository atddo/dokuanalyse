import itertools
import numpy as np
from corpus import CorpusLoader
from evaluation import CrossValidation, ClassificationEvaluator
from features import WordListNormalizer, BagOfWords, RelativeTermFrequencies
from classification import KNNClassifier
from visualization import bar_plot


def aufgabe3():

    # ********************************** ACHTUNG **************************************
    # Die nun zu implementierenden Funktionen spielen eine zentrale Rolle im weiteren
    # Verlauf des Fachprojekts. Achten Sie auf eine effiziente und 'saubere' Umsetzung.
    # Verwenden Sie geeignete Datenstrukturen und passende Python Funktionen.
    # Wenn Ihnen Ihr Ansatz sehr aufwaendig vorkommt, haben Sie vermutlich nicht die
    # passenden Datenstrukturen / Algorithmen / (highlevel) Python / NumPy Funktionen
    # verwendet. Fragen Sie in diesem Fall!
    #
    # Schauen Sie sich jetzt schon gruendlich die Klassen und deren Interfaces in den
    # mitgelieferten Modulen an. Wenn Sie Ihre Datenstrukturen von Anfang an dazu
    # passend waehlen, erleichtert dies deren spaetere Benutzung. Zusaetzlich bieten
    # diese Klassen bereits etwas Inspiration fuer Python-typisches Design, wie zum
    # Beispiel Duck-Typing.
    #
    # Zu einigen der vorgebenen Intefaces finden Sie Unit Tests in dem Paket 'test'.
    # Diese sind sehr hilfreich um zu ueberpruefen, ob ihre Implementierung zusammen
    # mit anderen mitgelieferten Implementierungen / Interfaces funktionieren wird.
    # Stellen Sie immer sicher, dass die Unit tests fuer die von Ihnen verwendeten
    # Funktionen erfolgreich sind.
    # Hinweis: Im Verlauf des Fachprojekts werden die Unit Tests nach und nach erfolg-
    # reich sein. Falls es sie zu Beginn stoert, wenn einzelne Unit Tests fehlschlagen
    # koennen Sie diese durch einen 'decorator' vor der Methodendefinition voruebergehend
    # abschalten: @unittest.skip('')
    # https://docs.python.org/2/library/unittest.html#skipping-tests-and-expected-failures
    # Denken Sie aber daran sie spaeter wieder zu aktivieren.
    #
    # Wenn etwas unklar ist, fragen Sie!
    # *********************************************************************************

    CorpusLoader.load()
    brown = CorpusLoader.brown_corpus()

    # Um eine willkuerliche Aufteilung der Daten in Training und Test zu vermeiden,
    # (machen Sie sich bewusst warum das problematisch ist)
    # verwendet man zur Evaluierung von Klassifikatoren eine Kreuzvalidierung.
    # Dabei wird der gesamte Datensatz in k disjunkte Ausschnitte (Folds) aufgeteilt.
    # Jeder dieser Ausschnitte wird einmal als Test Datensatz verwendet, waehrend alle
    # anderen k-1 Ausschnitte als Trainings Datensatz verwendet werden. Man erhaehlt also
    # k Gesamtfehlerraten und k klassenspezifische Fehlerraten ide man jeweils zu einer
    # gemeinsamen Fehlerrate fuer die gesamte Kreuzvalidierung mittelt. Beachten Sie,
    # dass dabei ein gewichtetes Mittel gebildet werden muss, da die einzelnen Test Folds
    # nicht unbedingt gleich gross sein muessen.

    # Fuehren Sie aufbauend auf den Ergebnissen aus aufgabe2 eine 5-Fold Kreuzvalidierung
    # fuer den k-Naechste-Nachbarn Klassifikator auf dem Brown Corpus durch. Dazu koennen
    # Sie die Klasse CrossValidation im evaluation Modul verwenden.
    #
    # Vollziehen Sie dazu nach wie die Klasse die Daten in Trainging und Test Folds aufteilt.
    # Fertigen Sie zu dem Schema eine Skizze an. Diskutieren Sie Vorteile und Nachteile.
    # Schauen Sie sich an, wie die eigentliche Kreuzvalidierung funktioniert. Erklaeren Sie
    # wie das Prinzip des Duck-Typing hier angewendet wird.
    #
    # Hinweise:
    #
    # Die Klasse CrossValidator verwendet die Klasse , die Sie schon
    # fuer aufgabe2 implementieren sollten. Kontrollieren Sie Ihre Umsetzung im Sinne der
    # Verwendung im CrossValidator.
    #
    # Fuer das Verstaendnis der Implementierung der Klasse CrossValidator ist der Eclipse-
    # Debugger sehr hilfreich.
    

        
    normalized_words = WordListNormalizer().normalize_words(brown.words())[1]
    vocabulary = BagOfWords.most_freq_words(normalized_words, 500)
    word_bag = BagOfWords(vocabulary)

    bow_mat = {}
    for cat in brown.categories():
        bow_mat[cat] = [WordListNormalizer().normalize_words(brown.words(fileids=doc))[1] for doc in brown.fileids(categories=cat)]
    category_dic = word_bag.category_bow_dict(bow_mat)

    cross_validator = CrossValidation(category_dic, 5)

    first_fold = cross_validator.corpus_fold(0)
    second_fold = cross_validator.corpus_fold(1)
    third_fold = cross_validator.corpus_fold(2)
    fourth_fold = cross_validator.corpus_fold(3)
    fifth_fold = cross_validator.corpus_fold(4)

    classificator = KNNClassifier(1, 'euclidean')

    classificator.estimate(first_fold[0], first_fold[1])
    result_first_fold = classificator.classify(first_fold[2])

    classificator.estimate(second_fold[0], second_fold[1])
    result_second_fold = classificator.classify(second_fold[2])

    classificator.estimate(third_fold[0], third_fold[1])
    result_third_fold = classificator.classify(third_fold[2])

    classificator.estimate(fourth_fold[0], fourth_fold[1])
    result_fourth_fold = classificator.classify(fourth_fold[2])

    classificator.estimate(fifth_fold[0], fifth_fold[1])
    result_fifth_fold = classificator.classify(fifth_fold[2])

    eval_first_fold = ClassificationEvaluator(result_first_fold, first_fold[3])
    eval_second_fold = ClassificationEvaluator(result_second_fold, second_fold[3])
    eval_third_fold = ClassificationEvaluator(result_third_fold, third_fold[3])
    eval_fourth_fold = ClassificationEvaluator(result_fourth_fold, fourth_fold[3])
    eval_fifth_fold = ClassificationEvaluator(result_fifth_fold, fifth_fold[3])

    list_eval_folds =[]
    list_eval_folds.append(eval_first_fold.error_rate()[0])
    list_eval_folds.append(eval_second_fold.error_rate()[0])
    list_eval_folds.append(eval_third_fold.error_rate()[0])
    list_eval_folds.append(eval_fourth_fold.error_rate()[0])
    list_eval_folds.append(eval_fifth_fold.error_rate()[0])
    np_array_eval_folds = np.array(list_eval_folds)
    print np_array_eval_folds
    mittelwert = np.mean(np_array_eval_folds)
    print "Mittelwert"
    print mittelwert
    print "++++++++++++"
    print "Minimum"
    print np.min(np_array_eval_folds, axis=0)

    #gewichtetes mittel fehlt noch...




    # Bag-of-Words Weighting
    #
    # Bisher enthalten die Bag-of-Words Histogramme absolute Frequenzen.
    # Dadurch sind die Repraesentationen abhaengig von der absoluten Anzahl
    # von Woertern in den Dokumenten.
    # Dies kann vermieden werden, in dem man die Bag-of-Words Histogramme mit
    # einem Normalisierungsfaktor gewichtet.
    #
    # Normalisieren Sie die Bag-of-Words Histogramme so, dass relative Frequenzen
    # verwendet werden. Implementieren und verwenden Sie die Klasse RelativeTermFrequencies
    # im features Modul.
    #
    # Wie erklaeren Sie das Ergebnis? Schauen Sie sich dazu noch einmal die
    # mittelere Anzahl von Woertern pro Dokument an (aufgabe2).
    #
    # Wie in der Literatur ueblich, verwenden wir den
    # Begriff des "Term". Ein Term bezeichnet ein Wort aus dem Vokabular ueber
    # dem die Bag-of-Words Histogramme gebildet werden. Ein Bag-of-Words Histogramm
    # wird daher auch als Term-Vektor bezeichnet.

    #normalized_words = WordListNormalizer().normalize_words(brown.words())[1]
    #vocab = BagOfWords.most_freq_words(normalized_words, 20)
    #word_bag = BagOfWords(vocab)

    word_dic_category = {}
    for cat in brown.categories():
        word_dic_category[cat] = [WordListNormalizer().normalize_words(brown.words(fileids = doc))[1] for doc in brown.fileids(categories=cat)]
    category_word_dic = word_bag.category_bow_dict(word_dic_category)

    category_weighted = {}
    for key in category_word_dic:
        category_weighted[key] = RelativeTermFrequencies.weighting(category_word_dic[key])

    # Zusaetzlich kann man noch die inverse Frequenz von Dokumenten beruecksichtigen
    # in denen ein bestimmter Term vorkommt. Diese Normalisierung wird als
    # inverse document frequency bezeichnet. Die Idee dahinter ist Woerter die in
    # vielen Dokumenten vorkommen weniger stark im Bag-of-Words Histogramm zu gewichten.
    # Die zugrundeliegende Annahme ist aehnlich wie bei den stopwords (aufgabe1), dass
    # Woerter, die in vielen Dokumenten vorkommen, weniger Bedeutung fuer die
    # Unterscheidung von Dokumenten in verschiedene Klassen / Kategorien haben als
    # Woerter, die nur in wenigen Dokumenten vorkommen.
    # Diese Gewichtung laesst sich statistisch aus den Beispieldaten ermitteln.
    #
    # Zusammen mit der relativen Term Gewichtung ergibt sich die so genannte
    # "term frequency inverse document frequency"
    #
    #                            Anzahl von term in document                       Anzahl Dokumente
    # tfidf( term, document )  = ----------------------------   x   log ( ---------------------------------- )
    #                             Anzahl Woerter in document              Anzahl Dokumente die term enthalten
    #
    # http://www.tfidf.com
    #
    # Eklaeren Sie die Formel. Plotten Sie die inverse document frequency fuer jeden
    # Term ueber dem Brown Corpus.
    #
    # Implementieren und verwenden Sie die Klasse RelativeInverseDocumentWordFrequecies
    # im features Modul, in der Sie ein tfidf Gewichtungsschema umsetzen.
    # Ermitteln Sie die Gesamt- und klassenspezifischen Fehlerraten mit der Kreuzvalidierung.
    # Vergleichen Sie das Ergebnis mit der absolten und relativen Gewichtung.
    # Erklaeren Sie die Unterschiede in den klassenspezifischen Fehlerraten. Schauen Sie
    # sich dazu die Verteilungen der Anzahl Woerter und Dokumente je Kategorie aus aufgabe1
    # an. In wie weit ist eine Interpretation moeglich?

#    raise NotImplementedError('Implement me')


    # Evaluieren Sie die beste Klassifikationsleistung
    #
    # Ermitteln Sie nun die Parameter fuer die bester Klassifikationsleistung des
    # k-naechste-Nachbarn Klassifikators auf dem Brown Corpus mit der Kreuzvalidierung.
    # Dabei wird gleichzeitig immer nur ein Parameter veraendert. Man hat eine lokal
    # optimale Parameterkonfiguration gefunden, wenn jede Aenderung eines Parameters
    # zu einer Verschlechterung der Fehlerrate fuehrt.
    #
    # Erlaeutern Sie warum eine solche Parameterkonfiguration lokal optimal ist.
    #
    # Testen Sie mindestens die angegebenen Werte fuer die folgenden Parameter:
    # 1. Groesse des Vokabulars typischer Woerter (100, 500, 1000, 2000!)
    # 2. Gewichtung der Bag-of-Words Histogramme (absolute, relative!, relative with inverse document frequency)
    # 3. Distanzfunktion fuer die Bestimmung der naechsten Nachbarn (Cityblock!, Euclidean, Cosine)
    # 4. Anzahl der betrachteten naechsten Nachbarn (1, 2, 3, 4, 5, 6!)
    #
    # Erklaeren Sie den Effekt aller Parameter.
    #
    # Erklaeren Sie den Effekt zwischen Gewichtungsschema und Distanzfunktion.
    print "+++++++++++++++++++++++++++++++++++++++++++++++"
    print "Testfunktion"
    
    vocab_size = 2000
    distance_function="cityblock"
    knn=6
    print "vocabsize:"  + vocab_size +"; distance" + distance_function +"; knn" + knn
    vocabulary = BagOfWords.most_freq_words(normalized_words, vocab_size)
    word_bag = BagOfWords(vocabulary)

    bow_mat = {}
    for cat in brown.categories():
        bow_mat[cat] = [WordListNormalizer().normalize_words(brown.words(fileids=doc))[1] for doc in brown.fileids(categories=cat)]
    category_dic = word_bag.category_bow_dict(bow_mat)
    relative_category_dict = RelativeTermFrequencies.weighting(bow_mat)
    cross_validator = CrossValidation(relative_category_dict, 5)

    first_fold = cross_validator.corpus_fold(0)
    second_fold = cross_validator.corpus_fold(1)
    third_fold = cross_validator.corpus_fold(2)
    fourth_fold = cross_validator.corpus_fold(3)
    fifth_fold = cross_validator.corpus_fold(4)

    classificator = KNNClassifier(knn, distance_function)

    classificator.estimate(first_fold[0], first_fold[1])
    result_first_fold = classificator.classify(first_fold[2])

    classificator.estimate(second_fold[0], second_fold[1])
    result_second_fold = classificator.classify(second_fold[2])

    classificator.estimate(third_fold[0], third_fold[1])
    result_third_fold = classificator.classify(third_fold[2])

    classificator.estimate(fourth_fold[0], fourth_fold[1])
    result_fourth_fold = classificator.classify(fourth_fold[2])

    classificator.estimate(fifth_fold[0], fifth_fold[1])
    result_fifth_fold = classificator.classify(fifth_fold[2])

    eval_first_fold = ClassificationEvaluator(result_first_fold, first_fold[3])
    eval_second_fold = ClassificationEvaluator(result_second_fold, second_fold[3])
    eval_third_fold = ClassificationEvaluator(result_third_fold, third_fold[3])
    eval_fourth_fold = ClassificationEvaluator(result_fourth_fold, fourth_fold[3])
    eval_fifth_fold = ClassificationEvaluator(result_fifth_fold, fifth_fold[3])

    list_eval_folds =[]
    list_eval_folds.append(eval_first_fold.error_rate()[0])
    list_eval_folds.append(eval_second_fold.error_rate()[0])
    list_eval_folds.append(eval_third_fold.error_rate()[0])
    list_eval_folds.append(eval_fourth_fold.error_rate()[0])
    list_eval_folds.append(eval_fifth_fold.error_rate()[0])
    np_array_eval_folds = np.array(list_eval_folds)
    print np_array_eval_folds
    mittelwert = np.mean(np_array_eval_folds)
    print "Mittelwert"
    print mittelwert
    print "++++++++++++"
    print "Minimum"
    print np.min(np_array_eval_folds, axis=0)


if __name__ == '__main__':
    aufgabe3()
