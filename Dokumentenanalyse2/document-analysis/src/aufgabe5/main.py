from corpus import CorpusLoader

def aufgabe5():
    
    #
    # Naive Bayes Klassifikation
    #
    # Der Naive Bayes Klassifikator basiert auf dem Satz von Bayes fuer bedingte
    # Wahrscheinlichkeiten:
    #
    #                   P( K_i ) P( f | K_i )
    # P( K_i | f ) =  -----------------------
    #                           P( f )
    #
    # Man kann also gewissermassen eine bedingte Wahrscheinlichkeit umdrehen,
    # wenn man die Einzelwahrscheinlichkeiten kennt. 
    #
    # Der Bayes Klassifikator ordnet nun einem Sample f eine Klasse K_i nach
    # maximaler a-posteriori Wahrscheinlichkeit zu. Der Index k, der dem Sample
    # zugeordneten Klasse bestimmt sich gemaess:
    # 
    # k = argmax P( K_i | f )
    #       i
    #
    # Der Klassifikator ist damit durch die Verteilungen von P( K_i ) und P( f | K_i )
    # definiert.
    # 
    # FRAGEN:
    # Warum wird die Verteilung zu P( f ) nicht benoetigt?
    #
    # Wie laesst sich die Verteilung zu P( K_i ) interpretieren?
    # 
    # Wie laesst sich die Verteilung P( f | K_i ) interpretieren?
    #
    # Welche Verteilungsmodelle nehmen wir hier fuer P( K_i ) und P(f | K_i ) an und
    # warum? In unserem Fall ist f eine Bag-of-Words Repraesentation eines Dokuments
    # und K_i eine Dokumentenkategorie.
    # 
    # Bei der direkten Umsetzung kommt es zu numerischen Problemen durch
    # zu kleinen Zahlen. Wodurch werden diese Probleme verursacht?
    # Loesen lassen sie sich durch eine Betrachtung der logarithmischen a-posteriori
    # Wahrscheinlichkeit. 
    # 
    # Warum? Erklaeren Sie ausserdem warum sich das Ergebnis der Klassifikation 
    # durch die logarithmische Betrachtung nicht aendert.
    #
    # Stellen Sie die entsprechende Formel unter Beruecksichtigung der Verteilungsmodelle um. 
    #
    # Wie lassen sich die Verteilungen aus Beispieldaten statistisch schaetzen?
    #
    
    
    CorpusLoader.load()
    brown = CorpusLoader.brown_corpus()
    
    #
    # Implementieren Sie den Bayes Klassifikator und integrieren Sie ihn in die
    # Kreuzvalidierung zur Kategorisierung der Dokumente des Brown Corpus (classification Modul).
    # Fuehren Sie die Vorverarbeitung und Merkmalsberechnung wie bei dem K Naechste 
    # Nachbarn Klassifikator durch.
    # Evaluieren Sie die Fehlerrate fuer verschiedene Groessen des Vokabulars (500, 1000, 1500).
    # Lassen sich die verschiedenen Term Gewichtungen sinnvoll einsetzen (absolut, relativ, idf)?
    # Fuehren Sie gegebenenfalls eine entsprechende Evaluierung durch.
    #
    # Vergleichen Sie die Fehlerraten mit denen des K Naechste Nachbarn Klassifikators.
    #
    
    raise NotImplementedError('Implement me')

if __name__ == '__main__':
    aufgabe5()
