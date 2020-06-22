## Feature Extractor

Il feature extractor è un componente essenziale della maggior parte dei modelli di classificazione basati su reti neurali; il suo compito è, come suggerisce il nome stesso, quello di estrarre una serie di caratteristiche che "riassumano" i tratti salienti dell'oggetto passato in ingresso, che nel nostro caso è un'immagine RGB proveniente da una camera fissa. 

Nell'ambito della Computer Vision, esistono una serie di feature extractor standard che vengono utilizzati per un vasto numero di teask, in quanto risultano molto flessibili e in grado di fornire feature di alto livello che possono venire in contro alle esigenze di problemi anche molto diversi tra loro.

!!!!!!!!!! CITARE !!!!!!!!!!!!!!

<br>

| ![](https://github.com/FabioLanzi/PyNowCast/blob/master/resources/nowcast_fx.jpg) |
| -------------- |
| **Figura 1.1.** Schema a blocchi dell'autoencoder utilizzato durante la fase di allenamento del feature extractor; il feature extractor vero e proprio è rappresentato dalla sola parte di encoding (ramo "blu" in figura).     |

<br>

Nel nostro caso specifico, tuttavia, risulta più opportuno affidarsi ad un feature extractor più mirato e incentrato in sulle nostre esigenze particolari. Il problema che _PyNowCast_ va ad affrontare infatti è molto vincolato e i vincoli che lo contraddistinguono, se ben sfruttati, possono semplificarlo enormemente. In questo senso, per il nostro package abbiamo deciso di orientarci verso un feature extractor personalizzato basato su un autoencoder, che sia in grado di essere allenato efficacemente in modo non supervisionato.

Per adottare questa soluzione ci siamo basati su una semplice osservazione: poiché ci troviamo a trattare immagini provenienti da una camera fissa che riprende una certa porzione di paesaggio, gli elementi che variano tra un'immagine e l'altra sono essenzilamente la condizione metereologica e le condizioni di illuminazione. Fortunatamente ciò che varia sono esattamente le feature che servono ad un classificatore che si occupa di nowcasting. 

In tal senso, l'uso di un autoencoder composto da un encoder e un decoder speculari risulta particolarmente appropriato. Allenando un autoencoder di questo tipo a ricostruire semplicmente le immagini di input all'uscita dell'encoder si avrà un "codice" che rappresneta per l'appunto un riassunto delle immagini di input. Trovando la giusta dimensione di tale codice, l'encoder sarà forzato a rimuovere tutte le informazioni ridondati, che nel nostro caso sono appunto le caratteristiche che non variano tra un'immagine e l'altra; al contempo dovrà preservare gli elementi mutevoli delle medesime (meteo e condizioni di illuminazione).