Monetizacija:
​

Licenciranje softvera: Prodaja licenci tvrtkama, školama, organizatorima događanja za korištenje sustava.​

Pretplate: Mjesečne ili godišnje pretplate za pristup naprednim funkcijama i analitici.​

Freemium model: Besplatan osnovni paket s opcijama za nadogradnju na premium funkcionalnosti.​

Personalizirane usluge: Ponuda prilagodbe sustava prema specifičnim potrebama korisnika (npr. integracija s postojećim sustavima).​

Brendirani paketi: Ponuda specijaliziranih paketa za događanja ili korporacije s personaliziranim rješenjima.​

Oglašavanje unutar aplikacije: Dodavanje oglasa ili partnerskih promocija u mobilnoj aplikaciji ili web sučelju.​

PREDNOSTI:
Evo nekoliko ključnih natuknica koje pokazuju zašto je *FaceMark* bolje od klasičnih načina bilježenja prisutnosti poput potpisa, QR kodova i sličnih metoda:

1. **Brzina i Efikasnost:**
   - Prepoznavanje lica omogućuje automatsko bilježenje prisutnosti u stvarnom vremenu, što eliminira potrebe za ručnim unosima ili skeniranjem QR kodova.
   - **Brzo prepoznavanje** – sustav identifikira korisnike gotovo odmah, bez čekanja na skeniranje ili čekanje na potpis.

2. **Bez Grešaka i Manipulacija:**
   - Nema prostora za pogreške poput pogrešnog potpisa ili skeniranja pogrešnog QR koda.
   - Tehnologija prepoznavanja lica eliminira rizik od **lažiranja prisutnosti** (npr. netko se potpiše ili koristi tuđi QR kod).

3. **Automatska Analiza i Izvještaji:**
   - Uz *FaceMark*, ne morate ručno bilježiti ili pratiti prisutnost. Sustav automatski prikuplja i analizira podatke, pružajući vam statističke uvide o prisutnosti.
   - **Detaljna analiza** može obuhvatiti obrasce dolazaka, kašnjenja i druge relevantne informacije bez potrebe za dodatnim unosima.

4. **Ušteda vremena i resursa:**
   - Nema potrebe za fizičkim upisivanjem, skeniranjem ili razmjenom papira, što štedi dragocjeno vrijeme i smanjuje potrebu za zaposlenicima koji se bave administrativnim zadacima.
   - **Brža implementacija** – dodavanje novih korisnika u sustav je gotovo instant, u usporedbi s ručnim unosima ili ručnim ažuriranjem QR kodova.

5. **Veća Sigurnost:**
   - Prepoznavanje lica omogućuje visoku razinu sigurnosti jer je teško manipulisati ili lažirati identitet putem tehnologije prepoznavanja.
   - **Nema zaboravljenih lozinki** ili problema s lažnim prijavama koje su česte kod sustava temeljenih na QR kodovima ili potpisima.

6. **Lakoća u Skaliranju:**
   - Korištenje tehnologije prepoznavanja lica olakšava primjenu u velikim okruženjima (škole, tvrtke, događanja) bez potrebe za dodatnim resursima za održavanje ručnih evidencija ili fizičkih sustava (kao što su papiri i QR kodovi).

7. **Kompatibilnost i Prilagodljivost:**
   - Sustav je kompatibilan sa svim računalima i mobilnim uređajima, što znači da se može brzo implementirati u bilo kojem okruženju.
   - **Fleksibilnost** omogućava integraciju s postojećim sustavima bez potrebe za kompleksnim ažuriranjima infrastrukture.

Ove prednosti jasno pokazuju da *FaceMark* nudi brže, sigurnije, učinkovitije i skalabilnije rješenje za praćenje prisutnosti u usporedbi s tradicionalnim metodama.



UI/UX STUFF
Intuitivno sučelje:

Jednostavno i jasno korisničko sučelje omogućava lako korištenje čak i korisnicima bez tehničkog iskustva.

Brza navigacija između funkcionalnosti bez potrebe za dodatnim objašnjenjima.

Automatizirano prepoznavanje lica:

Trenutno prepoznavanje lica za automatsko bilježenje prisutnosti u stvarnom vremenu.

Visoka preciznost i brzina prepoznavanja, čak i u gužvama ili s različitim kutovima gledanja kamere.

Analitika i izvještaji u stvarnom vremenu:

Detaljna statistika o prisutnosti, kašnjenju, učestalosti dolazaka i drugim relevantnim metrikama.

Automatski generirani izvještaji spremni za pregled i daljnju analizu.

Brzo dodavanje novih korisnika:

Novi korisnici mogu biti dodani u sustav u svega nekoliko sekundi, što omogućava brzu implementaciju.

Jednostavno dodavanje korisničkih profila s potrebnim podacima i slikama za prepoznavanje.

Kompatibilnost s uređajima:

Kompatibilan s računalima i mobilnim uređajima, omogućujući upotrebu u različitim okruženjima.

Laka integracija u postojeće sustave bez potrebe za dodatnim hardverskim ulaganjima.

Sigurnost podataka:

Zaštita podataka temelji se na naprednim sigurnosnim standardima, uz poštivanje privatnosti korisnika.

Lica nisu pohranjena, čime je osigurana privatnost i sigurnost korisničkih podataka.

Mobilna verzija:

Dostupna mobilna aplikacija za praćenje prisutnosti u realnom vremenu s jednostavnom integracijom kamere.

Funkcionalnosti mobilne verzije uključuju obavijesti, pregled statistika i promjenu postavki.

Vizualizacija podataka:

Grafički prikazi statistike i trendova prisutnosti u obliku dijagrama i grafova za lakše razumijevanje.

Mogućnost izvoza podataka u formate poput CSV za daljnju analizu.

Prilagodljive postavke:

Sučelje i funkcionalnosti mogu se prilagoditi specifičnim potrebama korisnika, uključujući veličinu baze i vrste izvještaja.

Integracija s drugim sustavima:

Mogućnost integracije s postojećim sustavima za evidenciju, sigurnost ili upravljanje zaposlenicima i događanjima.
# COMPUTER VISION ATTENDANCE MODULE (name still not definitive)  
Faculty of Informatics in Pula: [https://fipu.unipu.hr/](https://fipu.unipu.hr/)  

**Author:** Antonio Labinjan  
**Course:** Web Applications  
**Mentor:** doc.dr.sc. Nikola Tanković  

### **Brief Description:**  
An application for face recognition and attendance tracking (or other use cases) implemented using Flask, CLIP, and FAISS.

---

## **Features:**

### **Student:**

_No authentication is required to use these features._

- Save face embeddings via live feed  
- Save face embeddings by uploading local images  
- Mark attendance using face scanning  

---

### **Professor (Admin):**

- **Authentication**  
- Define courses with names and time intervals during which attendance can be marked  
- Add students to the system—each student will have a name and images for model recognition  
- Receive email notifications whenever a student successfully marks attendance  
- View and filter attendance records  
- Download attendance reports in CSV format  
- Share attendance reports via email  
- Delete attendance records  
- View statistics  
- View a list of all students  
- Visualize data through three (for now) different types of charts  
- Use an internal announcement forum for professors (CRUD operations on announcements)  
- View the attendance percentage for individual students in specific courses  
- Analyze student late arrivals  
- Access the official academic calendar  

---

## **Login Information:**  
It is recommended to create your own account with your email. However, if you prefer not to, you can use my credentials:

- **Username:** Antonio  
- **Password:** 4uGnsUh9!!!  
- **Email:** alabinjan6@gmail.com  

---

### **Demo Video:**  
Watch the demonstration on YouTube: [https://youtu.be/hQDcAjGRHMQ](https://youtu.be/hQDcAjGRHMQ)
