- sia $x$ l'input del singolo blocco residuale
- sia $f$ una funzione di attivazione (LeakyReLU con negative slope = 0.01)
- siano $\phi_1$ e  $\phi_2$ due differenti layer convolutivi
- definiamo  $y_1 = f\left( \phi_1(x) \right )$ e  $y_2 = \phi_2(x)$
- l'output $y$ del singolo blocco residuale è dato da  $y = f \left( \phi_1(x) + \phi_2(x) \right )$

