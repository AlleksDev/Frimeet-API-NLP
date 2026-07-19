"""Build reproducible bootstrap datasets for Places intent and retrieval.

The generated examples are synthetic and contain no user data.  They exercise
open-vocabulary categories, spelling variation, contextual slots, and hard
retrieval negatives.  They are intended as a reviewed seed corpus; production
behavioral data should be anonymized, adjudicated, and added before release.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

try:
    from scripts.train_place_intent_bert import validate_training_example
    from scripts.train_place_retriever import _read_training_rows
except ModuleNotFoundError:  # Supports ``python scripts/build_...py``.
    from train_place_intent_bert import validate_training_example
    from train_place_retriever import _read_training_rows


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "training"
INTENT_ROOT = DATA_ROOT / "places_intent_v1"
RETRIEVAL_ROOT = DATA_ROOT / "places_retrieval_v1"
SPLITS = ("train", "validation", "test")


INTENT_CONCEPTS: dict[str, tuple[str, ...]] = {
    "train": (
        "donas artesanales",
        "bubble tea",
        "salón de té",
        "tacos de birria",
        "ramen",
        "comida coreana",
        "restaurante vegano",
        "cafetería de libros",
        "espacio de coworking",
        "jardín botánico",
        "mirador panorámico",
        "sendero natural",
        "museo de ciencia",
        "galería de arte",
        "biblioteca pública",
        "cine independiente",
        "café de juegos de mesa",
        "arcade",
        "karaoke",
        "muro de escalada",
        "estudio de yoga",
        "mercado artesanal",
        "heladería italiana",
        "hostal juvenil",
        "bar con música en vivo",
        "cafetería de especialidad",
        "marisquería",
        "desayunos chiapanecos",
    ),
    "validation": (
        "panadería sin gluten",
        "restaurante etíope",
        "casa de jazz",
        "pista de patinaje",
        "chocolatería",
        "vivero de orquídeas",
    ),
    "test": (
        "tienda de cómics",
        "temazcal",
        "taller de cerámica",
        "restaurante de comida libanesa",
        "parque para perros",
        "observatorio astronómico",
    ),
}

# Controlled orthographic and colloquial variants stay inside the CATEGORY
# span. They teach boundary detection without mapping the phrase to a closed
# class, and every concept remains confined to one split.
INTENT_CATEGORY_VARIANTS = {
    "donas artesanales": "donas artensanales",
    "bubble tea": "buble tea",
    "salón de té": "salon de te",
    "tacos de birria": "tacos d birria",
    "cafetería de libros": "cafetría con libros",
    "espacio de coworking": "espacio de co-working",
    "jardín botánico": "jardin botanico",
    "museo de ciencia": "museo d ciencia",
    "café de juegos de mesa": "cafe con juegos de mesa",
    "heladería italiana": "eladería italiana",
    "cafetería de especialidad": "cafeteria de especialidad",
    "panadería sin gluten": "panaderia sin glúten",
    "restaurante etíope": "resturante etíope",
    "pista de patinaje": "pista para patinar",
    "tienda de cómics": "tienda de comics",
    "taller de cerámica": "tayer de ceramica",
    "restaurante de comida libanesa": "resturante libanés",
    "observatorio astronómico": "observatorio astronomico",
}

PREFERENCES = (
    "tranquilo",
    "con terraza",
    "pet friendly",
    "económico",
    "romántico",
    "con wifi",
    "familiar",
    "al aire libre",
    "accesible",
    "abierto de noche",
)
EXCLUSIONS = (
    "ruido",
    "filas largas",
    "música fuerte",
    "humo",
    "alcohol",
    "aglomeraciones",
)
LOCATIONS = (
    "Centro Histórico",
    "Parque Central",
    "Plaza del Sol",
    "Universidad Autónoma",
    "Barrio de Guadalupe",
    "Terminal de Autobuses",
)
RADII = ("500 m", "1 km", "2 km", "3 km", "750 metros")
REFERENCES = (
    "Casa Azul",
    "La Estación",
    "Punto Cero",
    "El Farol",
    "Café Nómada",
    "La Casona",
    "Patio Central",
    "Foro Once",
)


INTENT_EXTRAS: dict[str, tuple[tuple[str, tuple[tuple[str, str], ...]], ...]] = {
    "train": (
        ("Dame algunas opciones para salir.", ()),
        ("Quiero algo parecido a Casa Azul.", (("REFERENCE", "Casa Azul"),)),
        (
            "Algo como Café Nómada pero sin música.",
            (("REFERENCE", "Café Nómada"), ("EXCLUSION", "música")),
        ),
        ("Que sea silencioso y con wifi.", (("PREFERENCE", "silencioso"), ("PREFERENCE", "con wifi"))),
        ("No quiero lugares con humo.", (("EXCLUSION", "humo"),)),
        ("Cerca del Parque Bicentenario.", (("LOCATION", "Parque Bicentenario"),)),
        ("A no más de 3 km.", (("RADIUS", "3 km"),)),
        (
            "Parecido a La Estación cerca de Plaza Ámbar.",
            (("REFERENCE", "La Estación"), ("LOCATION", "Plaza Ámbar")),
        ),
    ),
    "validation": (
        ("Sorpréndeme con algún lugar distinto.", ()),
        ("Algo similar a Casa Frida.", (("REFERENCE", "Casa Frida"),)),
        ("Sin niños y sin música alta.", (("EXCLUSION", "niños"), ("EXCLUSION", "música alta"))),
        ("Que tenga enchufes y sea silencioso.", (("PREFERENCE", "enchufes"), ("PREFERENCE", "silencioso"))),
        ("Alrededor del Teatro de la Ciudad.", (("LOCATION", "Teatro de la Ciudad"),)),
        ("En un radio de 1200 metros.", (("RADIUS", "1200 metros"),)),
        (
            "Como El Farol pero cerca del Barrio Antiguo.",
            (("REFERENCE", "El Farol"), ("LOCATION", "Barrio Antiguo")),
        ),
        ("Busco algo para pasar la tarde.", ()),
    ),
    "test": (
        ("No sé qué quiero, enséñame lugares.", ()),
        ("Uno como La Casona.", (("REFERENCE", "La Casona"),)),
        ("Sin escaleras y sin demasiado ruido.", (("EXCLUSION", "escaleras"), ("EXCLUSION", "demasiado ruido"))),
        ("Que acepte mascotas y tenga estacionamiento.", (("PREFERENCE", "acepte mascotas"), ("PREFERENCE", "estacionamiento"))),
        ("Por la zona del Museo Regional.", (("LOCATION", "Museo Regional"),)),
        ("Máximo a 5 kilómetros.", (("RADIUS", "5 kilómetros"),)),
        (
            "Parecido a Punto Cero alrededor de la Glorieta de la Paz.",
            (("REFERENCE", "Punto Cero"), ("LOCATION", "Glorieta de la Paz")),
        ),
        ("Quiero conocer algo nuevo hoy.", ()),
    ),
}


@dataclass(frozen=True)
class PlaceSpec:
    split: str
    family: str
    name: str
    category: str
    description: str
    tags: str
    tag_families: str
    query: str
    concept_key: str | None = None


PLACE_SPECS: tuple[PlaceSpec, ...] = (
    PlaceSpec("train", "sweets", "Donas del Parque", "donut_shop", "Donas glaseadas y rellenas horneadas cada mañana.", "donas postres café", "comida ambiente", "se me antojan donas glaseadas"),
    PlaceSpec("train", "drinks", "Boba Nube", "bubble_tea_shop", "Tés fríos con tapioca, frutas y opciones sin lácteos.", "bubble tea tapioca bebidas", "comida preferencias", "dónde venden boba o té con tapioca"),
    PlaceSpec("train", "drinks", "Casa Té Jazmín", "tea_house", "Salón de té tranquilo con infusiones y repostería ligera.", "té infusiones tranquilo", "comida ambiente", "quiero un salón de té silencioso"),
    PlaceSpec("train", "food", "Birria del Barrio", "birria_restaurant", "Tacos y consomé de birria preparados al estilo tradicional.", "birria tacos consomé", "comida ocasión", "busco tacos de birria con consomé"),
    PlaceSpec("train", "food", "Ramen Neko", "ramen_restaurant", "Ramen japonés con caldos largos y alternativas vegetarianas.", "ramen japonés fideos", "comida preferencias", "quiero un buen ramen japonés"),
    PlaceSpec("train", "food", "Seúl Mesa", "korean_restaurant", "Cocina coreana con bibimbap, kimchi y parrilla de mesa.", "comida coreana kimchi bibimbap", "comida cultura", "dónde puedo comer comida coreana"),
    PlaceSpec("train", "food", "Verde Raíz", "vegan_restaurant", "Platillos veganos de temporada con ingredientes locales.", "vegano saludable local", "comida preferencias", "restaurante vegano con comida completa"),
    PlaceSpec("train", "work", "Café Página", "book_cafe", "Cafetería con libreros, mesas de lectura y ambiente calmado.", "libros café lectura", "actividad ambiente", "cafetería para leer entre libros"),
    PlaceSpec("train", "work", "Nube Cowork", "coworking_space", "Espacio de trabajo con internet rápido, cabinas y salas por hora.", "coworking wifi trabajo remoto", "actividad servicios", "necesito coworking con wifi y cabina"),
    PlaceSpec("train", "nature", "Jardín Ceiba", "botanical_garden", "Colección de plantas tropicales con senderos interpretativos.", "jardín botánico plantas naturaleza", "actividad ambiente", "quiero visitar un jardín botánico"),
    PlaceSpec("train", "nature", "Mirador del Cañón", "viewpoint", "Mirador al aire libre con vista panorámica del valle.", "mirador paisaje fotografía", "actividad ambiente", "un mirador para ver el atardecer"),
    PlaceSpec("train", "nature", "Sendero Encino", "nature_trail", "Ruta arbolada para caminata moderada y observación de aves.", "senderismo aves bosque", "actividad ambiente", "sendero natural para caminar"),
    PlaceSpec("train", "culture", "Museo Cosmos", "science_museum", "Exhibiciones interactivas de astronomía, física y tecnología.", "museo ciencia tecnología", "actividad cultura", "museo interactivo de ciencia"),
    PlaceSpec("train", "culture", "Galería Sur", "art_gallery", "Exposiciones temporales de artistas locales y arte contemporáneo.", "galería arte exposiciones", "actividad cultura", "galería de arte contemporáneo"),
    PlaceSpec("train", "culture", "Biblioteca Central", "public_library", "Acervo público, hemeroteca y salas silenciosas de estudio.", "biblioteca libros estudio", "actividad servicios", "biblioteca pública para estudiar"),
    PlaceSpec("train", "culture", "Cine Lumière", "independent_cinema", "Sala independiente con cine de autor y ciclos internacionales.", "cine independiente películas", "actividad cultura", "dónde proyectan cine independiente"),
    PlaceSpec("train", "entertainment", "Dados y Café", "board_game_cafe", "Juegos de mesa para grupos con bebidas y alimentos ligeros.", "juegos de mesa café grupos", "actividad ocasión", "café con juegos de mesa"),
    PlaceSpec("train", "entertainment", "Pixel Arcade", "arcade", "Máquinas arcade clásicas, simuladores y torneos casuales.", "arcade videojuegos retro", "actividad tecnología", "quiero jugar maquinitas arcade"),
    PlaceSpec("train", "entertainment", "Karaoke Luna", "karaoke_bar", "Salas privadas de karaoke con catálogo en español e inglés.", "karaoke música salas privadas", "actividad ocasión", "karaoke con sala privada"),
    PlaceSpec("train", "wellness", "Muro Alto", "climbing_gym", "Muros de escalada y boulder para principiantes y avanzados.", "escalada boulder ejercicio", "actividad deporte", "gimnasio con muro de escalada"),
    PlaceSpec("train", "wellness", "Yoga Amanecer", "yoga_studio", "Clases de yoga suave, restaurativo y meditación guiada.", "yoga meditación bienestar", "actividad ambiente", "estudio para clases de yoga"),
    PlaceSpec("train", "shopping", "Mercado Manos", "artisan_market", "Mercado de textiles, cerámica y productos de artesanos locales.", "artesanías textiles cerámica", "compras cultura", "mercado de artesanías locales"),
    PlaceSpec("train", "sweets", "Gelato Chiapas", "ice_cream_shop", "Helado italiano elaborado con frutas regionales.", "helado gelato postres", "comida preferencias", "heladería de gelato artesanal"),
    PlaceSpec("train", "lodging", "Hostal Viajero", "hostel", "Hospedaje económico con dormitorios y cocina compartida.", "hostal económico viajeros", "alojamiento servicios", "hostal barato para mochileros"),
    PlaceSpec("train", "nightlife", "Foro Sonoro", "live_music_bar", "Bar con escenario para bandas locales y conciertos pequeños.", "música en vivo bar conciertos", "actividad ambiente", "bar con música en vivo"),
    PlaceSpec("train", "drinks", "Café Altura", "specialty_coffee_shop", "Café de especialidad con granos regionales y métodos filtrados.", "café especialidad métodos", "comida preferencias", "cafetería de especialidad con buen espresso"),
    PlaceSpec("train", "food", "Mar Azul", "seafood_restaurant", "Mariscos frescos, ceviches y pescados a la plancha.", "mariscos ceviche pescado", "comida ocasión", "marisquería para comer ceviche"),
    PlaceSpec("train", "food", "Cocina Zoque", "regional_breakfast_restaurant", "Desayunos regionales con tamales, tascalate y huevos.", "desayuno chiapaneco regional", "comida cultura", "lugar de desayunos chiapanecos"),
    PlaceSpec("train", "sweets", "Pan Libre", "gluten_free_bakery", "Panadería dedicada a panes y postres sin gluten.", "panadería sin gluten postres", "comida preferencias", "panadería con opciones sin gluten"),
    PlaceSpec("train", "food", "Addis Mesa", "ethiopian_restaurant", "Cocina etíope para compartir con injera y guisos especiados.", "comida etíope injera", "comida cultura", "quiero probar injera y comida etíope"),
    PlaceSpec("train", "nightlife", "Jazz Sótano", "jazz_club", "Club íntimo con presentaciones de jazz y sesiones improvisadas.", "jazz música en vivo", "actividad ambiente", "casa de jazz con música en vivo"),
    PlaceSpec("train", "entertainment", "Rueda Libre", "skating_rink", "Pista cubierta para patines con renta de equipo.", "patinaje pista ejercicio", "actividad deporte", "pista para patinar bajo techo"),
    PlaceSpec("train", "sweets", "Cacao Taller", "chocolate_shop", "Chocolatería con cacao local, bebidas y bombones artesanales.", "chocolate cacao bombones", "comida cultura", "chocolatería artesanal con cacao local"),
    PlaceSpec("train", "shopping", "Vivero Orquídea", "plant_nursery", "Vivero especializado en orquídeas, plantas y asesoría.", "vivero orquídeas plantas", "compras naturaleza", "vivero donde vendan orquídeas"),
    PlaceSpec("train", "shopping", "Comic Planet", "comic_book_store", "Tienda de cómics, manga, novelas gráficas y coleccionables.", "cómics manga coleccionables", "compras cultura", "tienda de comics y manga"),
    PlaceSpec("train", "wellness", "Temazcal Tierra", "temazcal", "Baño de vapor tradicional con sesiones guiadas de bienestar.", "temazcal vapor bienestar", "actividad cultura", "quiero una sesión de temazcal"),
    PlaceSpec("train", "culture", "Barro Vivo", "pottery_studio", "Taller de cerámica con clases, torno y pintura de piezas.", "cerámica barro talleres", "actividad cultura", "taller para aprender cerámica"),
    PlaceSpec("train", "food", "Cedro Libanés", "lebanese_restaurant", "Cocina libanesa con hummus, falafel y pan recién hecho.", "comida libanesa hummus falafel", "comida cultura", "restaurante libanés con falafel"),
    PlaceSpec("train", "nature", "Huellas", "dog_park", "Parque cercado con áreas de juego y bebederos para perros.", "parque perros mascotas", "actividad ambiente", "parque cercado para llevar a mi perro"),
    PlaceSpec("train", "culture", "Observatorio Sierra", "astronomical_observatory", "Observación nocturna con telescopios y charlas de astronomía.", "observatorio estrellas telescopio", "actividad cultura", "observatorio para ver estrellas"),
    PlaceSpec("train", "food", "Taquería Nocturna", "taco_shop", "Tacos al carbón y guisos servidos hasta la madrugada.", "tacos nocturno económico", "comida ambiente", "taquería abierta de madrugada"),
    PlaceSpec("train", "nightlife", "Agave Casa", "mezcal_bar", "Mezcalería con degustaciones guiadas de productores regionales.", "mezcal degustación agave", "comida cultura", "mezcalería con degustación"),
    PlaceSpec("train", "shopping", "Librería Andante", "bookstore", "Librería independiente con narrativa, ensayo y presentaciones.", "librería libros lectura", "compras cultura", "librería independiente con libros"),
    PlaceSpec("train", "culture", "Fototeca Luz", "photography_museum", "Archivo y exposiciones dedicadas a fotografía histórica.", "fotografía museo exposiciones", "actividad cultura", "museo dedicado a fotografía"),
    PlaceSpec("train", "entertainment", "Casa Salsa", "dance_studio", "Clases sociales de salsa, bachata y ritmos latinos.", "baile salsa bachata", "actividad deporte", "clases de salsa para principiantes"),
    PlaceSpec("train", "wellness", "Cancha Urbana", "sports_centre", "Canchas multiusos para básquetbol, voleibol y fútbol rápido.", "canchas deportes ejercicio", "actividad deporte", "centro deportivo con canchas"),
    PlaceSpec("train", "family", "Ludoteca Peques", "children_activity_centre", "Juegos educativos, lectura y actividades para niñas y niños.", "ludoteca niños juegos", "actividad ocasión", "ludoteca con actividades infantiles"),
    PlaceSpec("train", "nightlife", "Azotea 360", "rooftop_bar", "Terraza en azotea con coctelería y vista de la ciudad.", "rooftop terraza cocteles", "ambiente ocasión", "bar de azotea con vista"),
    PlaceSpec("validation", "food", "Pupusas Centroamérica", "pupusa_restaurant", "Pupusas de queso, frijol y chicharrón hechas al momento.", "pupusas comida salvadoreña", "comida cultura", "dónde comer pupusas salvadoreñas"),
    PlaceSpec("validation", "sweets", "Crêpe Maison", "creperie", "Crepas dulces y saladas con café y fruta fresca.", "crepas postres café", "comida ambiente", "crepería con opciones saladas"),
    PlaceSpec("validation", "culture", "Planetario Orión", "planetarium", "Proyecciones del cielo y actividades educativas de astronomía.", "planetario estrellas ciencia", "actividad cultura", "planetario con funciones de astronomía"),
    PlaceSpec("validation", "wellness", "Aqua Centro", "aquatic_centre", "Alberca techada con carriles y clases de natación.", "natación alberca ejercicio", "actividad deporte", "centro acuático con alberca techada"),
    PlaceSpec("validation", "entertainment", "Clave Oculta", "escape_room", "Salas de escape con acertijos para equipos pequeños.", "escape room acertijos grupos", "actividad ocasión", "quiero un escape room para amigos"),
    PlaceSpec("validation", "shopping", "Vinilo Sur", "record_store", "Discos de vinilo nuevos y usados de diversos géneros.", "vinilos discos música", "compras cultura", "tienda de discos de vinilo"),
    PlaceSpec("validation", "wellness", "Centro Presente", "meditation_centre", "Meditación guiada y prácticas de respiración en silencio.", "meditación respiración silencio", "actividad ambiente", "centro tranquilo para meditar"),
    PlaceSpec("validation", "drinks", "Lúpulo Local", "craft_brewery", "Cervecería artesanal con recorridos y degustación de temporada.", "cerveza artesanal degustación", "comida actividad", "cervecería artesanal con degustación"),
    PlaceSpec("validation", "food", "Sushi Viajero", "conveyor_belt_sushi", "Sushi servido en banda transportadora con piezas preparadas al momento.", "sushi banda japonesa", "comida cultura", "sushi de banda transportadora"),
    PlaceSpec("validation", "shopping", "Cero Residuo", "zero_waste_store", "Tienda a granel de alimentos y productos de limpieza reutilizables.", "granel ecológico reutilizable", "compras preferencias", "tienda sin empaques para comprar a granel"),
    PlaceSpec("validation", "entertainment", "Salto Alto", "trampoline_park", "Parque interior de trampolines con zonas para distintas edades.", "trampolines saltos familiar", "actividad deporte", "parque de trampolines bajo techo"),
    PlaceSpec("validation", "culture", "Teatro Bel Canto", "opera_house", "Recinto para ópera, recitales vocales y música de cámara.", "ópera teatro conciertos", "actividad cultura", "teatro donde presenten ópera"),
    PlaceSpec("validation", "nature", "Finca Aroma", "coffee_farm_tour", "Recorrido por cafetales con explicación del cultivo y tostado.", "café finca recorrido", "actividad cultura", "tour por una finca de café"),
    PlaceSpec("validation", "wellness", "Raqueta Sur", "squash_club", "Canchas de squash con renta de equipo y clases individuales.", "squash raqueta canchas", "actividad deporte", "club con canchas de squash"),
    PlaceSpec("validation", "work", "Fábrica Lab", "makerspace", "Taller compartido con impresoras 3D, electrónica y herramientas.", "makerspace impresión 3d electrónica", "actividad tecnología", "makerspace con impresora 3d"),
    PlaceSpec("validation", "culture", "Estudio Obturador", "photography_studio", "Estudio fotográfico con ciclorama, iluminación y renta por hora.", "fotografía estudio iluminación", "actividad servicios", "estudio de fotografía con ciclorama"),
    PlaceSpec("validation", "drinks", "Verso Café", "poetry_cafe", "Cafetería con lecturas de poesía y micrófono abierto.", "poesía café lectura", "actividad cultura", "café con lecturas de poesía"),
    PlaceSpec("validation", "wellness", "Baño Nube", "hammam_spa", "Baño de vapor tipo hammam con circuitos de relajación.", "hammam vapor spa", "actividad ambiente", "spa con baño tipo hammam"),
    PlaceSpec("validation", "shopping", "Cosecha Directa", "farmers_market", "Mercado semanal de frutas, verduras y productos de agricultores.", "mercado agricultores orgánico", "compras comida", "mercado de productores y agricultores"),
    PlaceSpec("validation", "food", "Taco Verde", "vegan_taco_shop", "Tacos veganos con guisos de setas, legumbres y salsas caseras.", "tacos veganos setas", "comida preferencias", "taquería completamente vegana"),
    PlaceSpec("test", "food", "Arepa Morena", "arepa_restaurant", "Arepas venezolanas rellenas y preparadas al momento.", "arepas comida venezolana", "comida cultura", "lugar de arepas venezolanas"),
    PlaceSpec("test", "drinks", "Michi Café", "cat_cafe", "Cafetería con zona separada para convivencia con gatos rescatados.", "gatos café mascotas", "ambiente ocasión", "cafetería donde pueda convivir con gatos"),
    PlaceSpec("test", "family", "Laboratorio Curioso", "children_science_centre", "Experimentos y talleres de ciencia para público infantil.", "ciencia niños experimentos", "actividad cultura", "talleres de ciencia para niños"),
    PlaceSpec("test", "nature", "Santuario Monarca", "butterfly_sanctuary", "Jardín de conservación y observación de mariposas.", "mariposas santuario naturaleza", "actividad ambiente", "santuario para observar mariposas"),
    PlaceSpec("test", "entertainment", "Risa Abierta", "comedy_club", "Escenario de stand up y noches de micrófono abierto.", "comedia stand up espectáculo", "actividad ambiente", "club con show de stand up"),
    PlaceSpec("test", "work", "Café Babel", "language_exchange_cafe", "Encuentros de conversación para practicar varios idiomas.", "idiomas intercambio café", "actividad cultura", "café con intercambio de idiomas"),
    PlaceSpec("test", "sweets", "Miga Verde", "vegan_bakery", "Pan dulce y pasteles veganos sin ingredientes animales.", "panadería vegana postres", "comida preferencias", "panadería con pasteles veganos"),
    PlaceSpec("test", "lodging", "Bosque Domo", "glamping", "Domos equipados para acampar con comodidad cerca del bosque.", "glamping camping naturaleza", "alojamiento ambiente", "glamping en domo cerca de naturaleza"),
    PlaceSpec("test", "food", "La Media Luna", "empanada_shop", "Empanadas horneadas con rellenos salados y dulces.", "empanadas horno comida", "comida ocasión", "lugar especializado en empanadas"),
    PlaceSpec("test", "nature", "Herpetario Verde", "reptile_house", "Centro de conservación con reptiles y visitas educativas.", "reptiles conservación educación", "actividad naturaleza", "herpetario para conocer reptiles"),
    PlaceSpec("test", "culture", "Museo del Barro", "ceramics_museum", "Colección histórica de alfarería y cerámica regional.", "museo cerámica alfarería", "actividad cultura", "museo sobre cerámica regional"),
    PlaceSpec("test", "wellness", "Pista Furia", "roller_derby_rink", "Pista y entrenamientos de roller derby para varios niveles.", "roller derby patines deporte", "actividad deporte", "dónde entrenar roller derby"),
    PlaceSpec("test", "work", "Voz Estudio", "podcast_studio", "Cabinas tratadas acústicamente para grabar y editar podcasts.", "podcast grabación cabina", "actividad tecnología", "estudio para grabar un podcast"),
    PlaceSpec("test", "drinks", "Invernadero Café", "indoor_garden_cafe", "Cafetería dentro de un jardín interior con abundantes plantas.", "café jardín interior plantas", "ambiente comida", "cafetería dentro de un invernadero"),
    PlaceSpec("test", "family", "Robot Peques", "robotics_workshop", "Talleres infantiles de robótica, programación y construcción.", "robótica niños programación", "actividad tecnología", "taller de robótica para niños"),
    PlaceSpec("test", "nature", "Remo Río", "kayak_rental", "Renta de kayaks con equipo de seguridad y rutas guiadas.", "kayak remo río", "actividad deporte", "renta de kayak con guía"),
    PlaceSpec("test", "shopping", "Quesería Sierra", "cheese_shop", "Tienda de quesos regionales con degustación y productos artesanales.", "quesos degustación artesanal", "compras comida", "tienda especializada en quesos"),
    PlaceSpec("test", "wellness", "Circo Aire", "circus_school", "Escuela de telas aéreas, acrobacia y equilibrio para principiantes.", "circo acrobacia telas", "actividad deporte", "escuela para aprender telas aéreas"),
    PlaceSpec("test", "nature", "Reserva Alas", "birdwatching_reserve", "Reserva natural con observatorios y recorridos para avistar aves.", "aves reserva observación", "actividad naturaleza", "reserva para observación de aves"),
    PlaceSpec("test", "nightlife", "Bar Cero", "alcohol_free_bar", "Bar nocturno de coctelería sin alcohol y música moderada.", "mocktails sin alcohol nocturno", "comida ambiente", "bar nocturno sin bebidas alcohólicas"),
)


def main() -> None:
    INTENT_ROOT.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_ROOT.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "license": "Proprietary - Frimeet internal training data",
        "provenance": "Synthetic bootstrap corpus; contains no production user data",
        "intent": {},
        "retrieval": {},
    }
    for split in SPLITS:
        intent_records = _intent_records(split)
        retrieval_records = _retrieval_records(split)
        intent_path = INTENT_ROOT / f"{split}.jsonl"
        retrieval_path = RETRIEVAL_ROOT / f"{split}.jsonl"
        _write_jsonl(intent_path, intent_records)
        _write_jsonl(retrieval_path, retrieval_records)
        manifest["intent"][split] = _file_manifest(intent_path, intent_records)  # type: ignore[index]
        manifest["retrieval"][split] = _file_manifest(retrieval_path, retrieval_records)  # type: ignore[index]

    _validate_cross_split_uniqueness()
    _validate_generated_files()
    (DATA_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _intent_records(split: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, concept in enumerate(INTENT_CONCEPTS[split]):
        concept_variant = INTENT_CATEGORY_VARIANTS.get(concept, concept)
        preference = PREFERENCES[index % len(PREFERENCES)]
        exclusion = EXCLUSIONS[index % len(EXCLUSIONS)]
        location = LOCATIONS[index % len(LOCATIONS)]
        radius = RADII[index % len(RADII)]
        records.extend(
            (
                _annotated(f"Busco {concept}.", (("CATEGORY", concept),)),
                _annotated(
                    f"Quiero {concept_variant} {preference}.",
                    (("CATEGORY", concept_variant), ("PREFERENCE", preference)),
                ),
                _annotated(
                    f"Recomiéndame {concept} sin {exclusion} cerca de {location}.",
                    (
                        ("CATEGORY", concept),
                        ("EXCLUSION", exclusion),
                        ("LOCATION", location),
                    ),
                ),
                _annotated(
                    f"¿Hay {concept_variant} a menos de {radius} de {location}?",
                    (
                        ("CATEGORY", concept_variant),
                        ("RADIUS", radius),
                        ("LOCATION", location),
                    ),
                ),
            )
        )
        reference_examples = 12 if split == "train" else 2
        if index < reference_examples:
            reference = REFERENCES[index % len(REFERENCES)]
            records.append(
                _annotated(
                    f"Busco {concept_variant} parecido a {reference}.",
                    (("CATEGORY", concept_variant), ("REFERENCE", reference)),
                )
            )
    records.extend(_annotated(text, spans) for text, spans in INTENT_EXTRAS[split])
    return records


def _annotated(
    text: str,
    annotations: Sequence[tuple[str, str]],
) -> dict[str, object]:
    search_from: dict[str, int] = {}
    spans: list[dict[str, object]] = []
    for slot, value in annotations:
        start = text.index(value, search_from.get(value, 0))
        end = start + len(value)
        search_from[value] = end
        spans.append({"start": start, "end": end, "slot": slot})
    payload: dict[str, object] = {
        "text": text,
        "spans": sorted(spans, key=lambda span: (int(span["start"]), int(span["end"]))),
    }
    validate_training_example(payload)
    return payload


def _retrieval_records(split: str) -> list[dict[str, object]]:
    specs = _expanded_place_specs(split)
    documents = {spec.name: _place_document(spec) for spec in specs}
    records: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        same_family = [
            candidate
            for candidate in specs
            if candidate.name != spec.name
            and candidate.family == spec.family
            and _concept_key(candidate) != _concept_key(spec)
        ]
        other_family = [
            candidate
            for candidate in specs
            if candidate.name != spec.name
            and candidate.family != spec.family
            and _concept_key(candidate) != _concept_key(spec)
        ]
        rotated = other_family[index % len(other_family) :] + other_family[: index % len(other_family)]
        negative_specs = _unique_specs((*same_family, *rotated))[:3]
        if len(negative_specs) < 3:
            raise RuntimeError(f"Not enough hard negatives for {spec.name}")
        records.append(
            {
                "query": spec.query,
                "positive": documents[spec.name],
                "hard_negatives": [documents[item.name] for item in negative_specs],
            }
        )
    return records


def _expanded_place_specs(split: str) -> list[PlaceSpec]:
    """Create unique, discriminative venue documents without split leakage."""

    expanded: list[PlaceSpec] = []
    base_specs = [item for item in PLACE_SPECS if item.split == split]
    variants = 3 if split == "train" else 1
    for index, spec in enumerate(base_specs):
        expanded.append(replace(spec, concept_key=spec.name))
        if variants >= 2:
            expanded.append(
                replace(
                    spec,
                    name=f"{spec.name} Norte",
                    description=(
                        f"{spec.description} Esta sede ofrece acceso sin escalones, "
                        "señalización clara y un ambiente tranquilo."
                    ),
                    tags=f"{spec.tags} accesible tranquilo",
                    query=(
                        f"{spec.query}; prefiero acceso sin escalones y ambiente tranquilo"
                    ),
                    concept_key=spec.name,
                )
            )
        if variants >= 3:
            expanded.append(
                replace(
                    spec,
                    name=f"{spec.name} Jardín",
                    description=(
                        f"{spec.description} Esta sede cuenta con terraza, mesas al "
                        "aire libre y alternativas de precio moderado."
                    ),
                    tags=f"{spec.tags} terraza aire libre económico",
                    query=(
                        f"{spec.query}; ideal si tiene terraza y precios moderados"
                    ),
                    concept_key=spec.name,
                )
            )
        # Sixteen extra train-only documents reach 160 training pairs while
        # validation/test keep exactly one document per unseen concept. This
        # avoids ambiguous evaluation qrels caused by sibling venues.
        if split == "train" and index < 16:
            expanded.append(
                replace(
                    spec,
                    name=f"{spec.name} Plaza",
                    description=(
                        f"{spec.description} Esta sede acepta reservaciones, tiene "
                        "área para grupos y mantiene horario extendido."
                    ),
                    tags=f"{spec.tags} reservaciones grupos horario extendido",
                    query=(
                        f"{spec.query}; necesito reservar para un grupo y llegar tarde"
                    ),
                    concept_key=spec.name,
                )
            )
    return expanded


def _concept_key(spec: PlaceSpec) -> str:
    return spec.concept_key or spec.name


def _place_document(spec: PlaceSpec) -> str:
    return (
        f"Nombre: {spec.name}. "
        f"Tipo registrado: {spec.category.replace('_', ' ')}. "
        f"Descripcion: {spec.description} "
        f"Etiquetas: {spec.tags}. "
        f"Familias de etiquetas: {spec.tag_families}."
    )


def _unique_specs(specs: Iterable[PlaceSpec]) -> list[PlaceSpec]:
    unique: list[PlaceSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.name not in seen:
            unique.append(spec)
            seen.add(spec.name)
    return unique


def _write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _file_manifest(
    path: Path,
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "records": len(records),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _validate_cross_split_uniqueness() -> None:
    intent_seen: dict[str, str] = {}
    retrieval_seen: dict[str, str] = {}
    positive_seen: dict[str, str] = {}
    for split in SPLITS:
        for record in _intent_records(split):
            text = str(record["text"]).casefold()
            if text in intent_seen:
                raise ValueError(f"Duplicate intent text in {intent_seen[text]} and {split}: {text}")
            intent_seen[text] = split
        for record in _retrieval_records(split):
            query = str(record["query"]).casefold()
            positive = str(record["positive"])
            if query in retrieval_seen:
                raise ValueError(f"Duplicate retrieval query in {retrieval_seen[query]} and {split}: {query}")
            if positive in positive_seen:
                raise ValueError(f"Duplicate positive document in {positive_seen[positive]} and {split}")
            retrieval_seen[query] = split
            positive_seen[positive] = split


def _validate_generated_files() -> None:
    for split in SPLITS:
        _read_training_rows(RETRIEVAL_ROOT / f"{split}.jsonl")


if __name__ == "__main__":
    main()
