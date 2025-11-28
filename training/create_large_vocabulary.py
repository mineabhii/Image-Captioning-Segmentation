#!/usr/bin/env python3
"""
Create a large vocabulary tokenizer with 5000+ words for better caption accuracy
"""

import os
import pickle
import sys
from tensorflow.keras.preprocessing.text import Tokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.captioning_model import create_captioning_model

def create_comprehensive_captions():
    """Create a comprehensive set of captions covering various scenarios"""
    
    # Base caption templates
    base_captions = [
        # People and activities
        "a person sitting on a bench in the park",
        "a young woman walking down the street",
        "an elderly man reading a book outdoors",
        "children playing soccer in the field",
        "a couple holding hands on the beach",
        "people enjoying a picnic in the garden",
        "a photographer taking pictures of nature",
        "tourists visiting a famous landmark",
        "a jogger running through the forest",
        "students studying under a tree",
        
        # Animals
        "a brown dog running in the meadow",
        "a black cat sleeping on a windowsill",
        "birds flying across the blue sky",
        "a horse grazing in the green pasture",
        "dolphins swimming in the ocean",
        "a butterfly landing on a colorful flower",
        "squirrels climbing up tall trees",
        "fish swimming in crystal clear water",
        "sheep resting in the countryside",
        "an eagle soaring high above mountains",
        
        # Nature and landscapes
        "a beautiful mountain landscape with snow",
        "rolling hills covered with green grass",
        "a peaceful lake surrounded by trees",
        "ocean waves crashing against rocky cliffs",
        "a dense forest with tall pine trees",
        "wildflowers blooming in a meadow",
        "a waterfall cascading down rocks",
        "desert dunes under a blazing sun",
        "northern lights dancing in the sky",
        "autumn leaves falling from branches",
        
        # Urban scenes
        "busy city streets filled with traffic",
        "tall skyscrapers reaching toward clouds",
        "a vintage car parked on cobblestones",
        "colorful street art on building walls",
        "people shopping in outdoor markets",
        "a bridge spanning across a wide river",
        "construction workers building new structures",
        "bicycles lined up at a rental station",
        "food trucks serving delicious meals",
        "street performers entertaining crowds",
        
        # Weather and seasons
        "heavy rain falling on city pavements",
        "bright sunshine illuminating landscapes",
        "thick fog covering mountain peaks",
        "fresh snow blanketing the countryside",
        "storm clouds gathering in the distance",
        "rainbow appearing after summer rain",
        "morning dew glistening on grass",
        "strong winds bending tree branches",
        "clear starry night sky overhead",
        "warm golden sunset over horizon",
        
        # Food and dining
        "fresh fruits displayed at market stalls",
        "homemade bread cooling on racks",
        "steaming coffee in ceramic cups",
        "colorful vegetables growing in gardens",
        "chefs preparing meals in restaurants",
        "families dining together outdoors",
        "wine bottles aging in cellars",
        "fishermen catching fresh seafood",
        "farmers harvesting ripe crops",
        "children eating ice cream cones",
        
        # Transportation
        "trains traveling through scenic valleys",
        "airplanes flying through fluffy clouds",
        "boats sailing on calm waters",
        "motorcycles cruising mountain roads",
        "buses transporting people downtown",
        "trucks delivering goods across country",
        "helicopters hovering above cities",
        "submarines exploring ocean depths",
        "hot air balloons floating overhead",
        "rockets launching into space",
        
        # Architecture and buildings
        "ancient castles perched on hilltops",
        "modern glass towers reflecting sunlight",
        "traditional wooden houses with gardens",
        "gothic cathedrals with intricate details",
        "rustic barns surrounded by farmland",
        "elegant mansions with manicured lawns",
        "cozy cottages nestled in forests",
        "industrial warehouses in urban areas",
        "historic monuments preserving culture",
        "futuristic buildings with unique designs",
        
        # Technology and objects
        "smartphones displaying colorful screens",
        "laptops open on wooden desks",
        "cameras capturing precious moments",
        "musical instruments creating beautiful sounds",
        "books stacked on library shelves",
        "paintings hanging in art galleries",
        "sculptures displayed in museums",
        "tools organized in workshops",
        "machines operating in factories",
        "robots assisting with daily tasks",
        
        # Sports and recreation
        "athletes competing in Olympic games",
        "soccer players kicking balls skillfully",
        "swimmers diving into crystal pools",
        "tennis players serving powerful shots",
        "cyclists racing through mountain trails",
        "golfers putting on manicured greens",
        "skiers descending snowy slopes",
        "surfers riding massive ocean waves",
        "climbers ascending rocky mountains",
        "runners crossing finish lines triumphantly"
    ]
    
    # Expand with variations and additional details
    expanded_captions = []
    
    # Add base captions
    expanded_captions.extend(base_captions)
    
    # Add color variations
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "silver", "golden", "crimson", "navy", "emerald", "violet", "turquoise", "maroon", "beige"]
    
    # Add size variations  
    sizes = ["tiny", "small", "medium", "large", "huge", "enormous", "massive", "miniature", "gigantic", "compact", "spacious", "narrow", "wide", "tall", "short", "thick", "thin"]
    
    # Add descriptive adjectives
    descriptors = ["beautiful", "stunning", "magnificent", "gorgeous", "elegant", "charming", "lovely", "amazing", "wonderful", "fantastic", "incredible", "spectacular", "breathtaking", "impressive", "remarkable", "extraordinary", "outstanding", "excellent", "perfect", "brilliant"]
    
    # Add emotions and moods
    emotions = ["happy", "peaceful", "joyful", "serene", "excited", "calm", "relaxed", "energetic", "content", "cheerful", "delighted", "satisfied", "pleased", "grateful", "optimistic", "confident", "comfortable", "secure"]
    
    # Add time references
    times = ["morning", "afternoon", "evening", "night", "dawn", "dusk", "noon", "midnight", "sunrise", "sunset", "daylight", "twilight", "nighttime", "daytime"]
    
    # Add weather conditions
    weather = ["sunny", "cloudy", "rainy", "snowy", "foggy", "windy", "stormy", "clear", "overcast", "humid", "dry", "warm", "cool", "hot", "cold", "mild", "pleasant", "harsh"]
    
    # Create variations with different combinations
    simple_objects = ["car", "house", "tree", "flower", "mountain", "building", "road", "bridge", "boat", "plane"]
    
    for obj in simple_objects:
        for color in colors[:10]:  # Use first 10 colors
            expanded_captions.append(f"a {color} {obj} in the distance")
            expanded_captions.append(f"beautiful {color} {obj} standing alone")
            
        for size in sizes[:8]:  # Use first 8 sizes
            expanded_captions.append(f"a {size} {obj} in the landscape")
            expanded_captions.append(f"several {size} {obj}s grouped together")
            
        for descriptor in descriptors[:15]:  # Use first 15 descriptors
            expanded_captions.append(f"a {descriptor} {obj} photograph")
            expanded_captions.append(f"{descriptor} view of {obj}")
    
    # Add scene descriptions with multiple elements
    scene_templates = [
        "a {adj1} {weather} day with {obj1} and {obj2}",
        "{color1} {obj1} next to {color2} {obj2} during {time}",
        "{size} {obj1} surrounded by {adj2} {obj2}s",
        "peaceful scene showing {obj1} and {obj2} in {weather} conditions",
        "{adj1} photograph of {obj1} with {obj2} in background"
    ]
    
    import random
    for template in scene_templates:
        for _ in range(50):  # Generate 50 variations per template
            caption = template.format(
                adj1=random.choice(descriptors),
                adj2=random.choice(descriptors), 
                color1=random.choice(colors),
                color2=random.choice(colors),
                obj1=random.choice(simple_objects),
                obj2=random.choice(simple_objects),
                size=random.choice(sizes),
                weather=random.choice(weather),
                time=random.choice(times)
            )
            expanded_captions.append(caption)
    
    # Add technical and detailed descriptions
    technical_terms = [
        "high resolution digital photograph",
        "professional camera shot with shallow depth of field",
        "wide angle landscape photography",
        "macro close-up detailed image",
        "aerial drone footage from above",
        "vintage film photograph with grain",
        "black and white artistic composition",
        "long exposure night photography",
        "portrait photography with natural lighting",
        "documentary style street photography"
    ]
    expanded_captions.extend(technical_terms)
    
    # Add location-specific descriptions
    locations = ["park", "beach", "forest", "city", "countryside", "mountains", "desert", "island", "valley", "plateau", "canyon", "meadow", "field", "garden", "backyard", "rooftop", "balcony", "terrace", "courtyard", "plaza"]
    
    for location in locations:
        for weather_cond in weather[:10]:
            expanded_captions.append(f"{weather_cond} day in the {location}")
            expanded_captions.append(f"people enjoying {weather_cond} weather in {location}")
        
        for time_period in times[:8]:
            expanded_captions.append(f"{time_period} scene at the {location}")
            expanded_captions.append(f"peaceful {time_period} in {location} setting")
    
    # Add action-oriented descriptions
    actions = ["walking", "running", "sitting", "standing", "lying", "jumping", "dancing", "playing", "working", "relaxing", "exploring", "climbing", "swimming", "flying", "driving", "cycling", "hiking", "camping", "fishing", "gardening"]
    
    for action in actions:
        for location in locations[:10]:
            expanded_captions.append(f"people {action} in the {location}")
            expanded_captions.append(f"person {action} peacefully in {location}")
    
    # Add more comprehensive word lists to increase vocabulary
    additional_words = [
        # Extended colors
        "azure", "crimson", "magenta", "turquoise", "violet", "indigo", "coral", "salmon", "lime", "olive", "navy", "teal", "maroon", "burgundy", "beige", "tan", "khaki", "lavender", "peach", "mint",
        
        # Extended adjectives
        "magnificent", "spectacular", "breathtaking", "stunning", "gorgeous", "elegant", "charming", "delightful", "wonderful", "fantastic", "incredible", "amazing", "outstanding", "excellent", "perfect", "brilliant", "remarkable", "extraordinary", "impressive", "marvelous",
        "peaceful", "serene", "tranquil", "calm", "quiet", "gentle", "soft", "smooth", "rough", "harsh", "sharp", "bright", "dim", "dark", "light", "heavy", "thick", "thin", "wide", "narrow",
        "ancient", "modern", "contemporary", "vintage", "classic", "traditional", "futuristic", "historic", "old", "new", "fresh", "stale", "clean", "dirty", "pure", "natural", "artificial", "synthetic", "organic", "wild",
        
        # Extended objects and things
        "sculpture", "statue", "monument", "fountain", "tower", "castle", "palace", "mansion", "cottage", "cabin", "tent", "shelter", "garage", "shed", "barn", "warehouse", "factory", "office", "shop", "store",
        "bicycle", "motorcycle", "truck", "bus", "train", "airplane", "helicopter", "boat", "ship", "yacht", "canoe", "kayak", "surfboard", "skateboard", "scooter", "wheelchair", "stroller", "cart", "wagon", "trailer",
        "camera", "phone", "computer", "laptop", "tablet", "television", "radio", "speaker", "microphone", "guitar", "piano", "violin", "drums", "flute", "trumpet", "saxophone", "harmonica", "accordion", "harp", "organ",
        
        # Extended animals
        "elephant", "giraffe", "lion", "tiger", "leopard", "cheetah", "zebra", "rhino", "hippo", "monkey", "gorilla", "chimpanzee", "bear", "wolf", "fox", "deer", "rabbit", "squirrel", "raccoon", "skunk",
        "eagle", "hawk", "owl", "parrot", "peacock", "penguin", "flamingo", "swan", "duck", "goose", "chicken", "rooster", "turkey", "pigeon", "sparrow", "robin", "cardinal", "hummingbird", "crane", "stork",
        "fish", "shark", "whale", "dolphin", "octopus", "jellyfish", "starfish", "crab", "lobster", "shrimp", "turtle", "snake", "lizard", "frog", "toad", "butterfly", "bee", "ant", "spider", "beetle",
        
        # Extended nature elements
        "mountain", "hill", "valley", "canyon", "cliff", "rock", "stone", "boulder", "pebble", "sand", "beach", "shore", "coast", "ocean", "sea", "lake", "river", "stream", "waterfall", "pond",
        "forest", "jungle", "woodland", "grove", "meadow", "field", "prairie", "grassland", "desert", "oasis", "island", "peninsula", "glacier", "iceberg", "volcano", "geyser", "cave", "cavern", "tunnel", "crater",
        "tree", "oak", "pine", "maple", "birch", "willow", "palm", "cedar", "fir", "spruce", "bush", "shrub", "hedge", "vine", "moss", "fern", "grass", "weed", "flower", "rose", "tulip", "daisy", "lily", "orchid",
        
        # Extended weather and atmosphere
        "sunshine", "sunlight", "daylight", "moonlight", "starlight", "shadow", "shade", "darkness", "brightness", "glow", "sparkle", "shimmer", "glitter", "reflection", "mirror", "crystal", "diamond", "jewel", "gem", "pearl",
        "cloud", "fog", "mist", "haze", "smoke", "steam", "vapor", "rain", "drizzle", "shower", "storm", "thunder", "lightning", "snow", "ice", "frost", "dew", "wind", "breeze", "gust",
        
        # Extended actions and verbs
        "walking", "running", "jogging", "sprinting", "hiking", "climbing", "jumping", "leaping", "dancing", "singing", "playing", "working", "studying", "reading", "writing", "drawing", "painting", "photographing", "filming", "recording",
        "eating", "drinking", "cooking", "baking", "gardening", "farming", "fishing", "hunting", "swimming", "diving", "surfing", "sailing", "rowing", "paddling", "cycling", "driving", "flying", "soaring", "gliding", "floating",
        "building", "constructing", "creating", "making", "crafting", "designing", "decorating", "cleaning", "washing", "polishing", "repairing", "fixing", "maintaining", "organizing", "arranging", "sorting", "collecting", "gathering", "picking", "harvesting",
        
        # Extended descriptive terms
        "enormous", "gigantic", "massive", "huge", "large", "big", "medium", "small", "tiny", "miniature", "microscopic", "colossal", "immense", "vast", "spacious", "roomy", "cramped", "compact", "dense", "sparse",
        "tall", "high", "elevated", "towering", "short", "low", "deep", "shallow", "long", "extended", "brief", "quick", "fast", "rapid", "slow", "gradual", "sudden", "immediate", "instant", "delayed",
        "straight", "curved", "bent", "twisted", "circular", "round", "square", "rectangular", "triangular", "oval", "diamond", "hexagonal", "octagonal", "spiral", "zigzag", "wavy", "bumpy", "smooth", "flat", "sloped",
        
        # Extended locations and places
        "city", "town", "village", "hamlet", "metropolis", "suburb", "neighborhood", "district", "area", "region", "zone", "territory", "country", "nation", "continent", "world", "universe", "space", "cosmos", "galaxy",
        "street", "road", "avenue", "boulevard", "lane", "alley", "path", "trail", "sidewalk", "pavement", "crosswalk", "intersection", "corner", "square", "plaza", "courtyard", "yard", "garden", "park", "playground",
        "school", "college", "university", "library", "museum", "gallery", "theater", "cinema", "restaurant", "cafe", "bar", "pub", "hotel", "motel", "inn", "resort", "spa", "gym", "stadium", "arena",
        
        # Extended time references
        "second", "minute", "hour", "day", "week", "month", "year", "decade", "century", "millennium", "moment", "instant", "period", "duration", "interval", "season", "spring", "summer", "autumn", "winter",
        "morning", "noon", "afternoon", "evening", "night", "midnight", "dawn", "sunrise", "sunset", "dusk", "twilight", "daybreak", "nightfall", "today", "yesterday", "tomorrow", "now", "then", "soon", "later",
        
        # Extended materials and textures
        "wood", "metal", "steel", "iron", "copper", "bronze", "gold", "silver", "aluminum", "plastic", "rubber", "leather", "fabric", "cotton", "silk", "wool", "linen", "canvas", "paper", "cardboard",
        "glass", "crystal", "ceramic", "porcelain", "clay", "mud", "dirt", "soil", "sand", "gravel", "concrete", "cement", "brick", "stone", "marble", "granite", "slate", "tile", "shingle", "plaster",
        
        # Extended professional and occupational terms
        "doctor", "nurse", "teacher", "student", "professor", "scientist", "researcher", "engineer", "architect", "designer", "artist", "musician", "writer", "journalist", "photographer", "filmmaker", "actor", "dancer", "athlete", "coach",
        "chef", "cook", "waiter", "bartender", "farmer", "gardener", "mechanic", "electrician", "plumber", "carpenter", "painter", "cleaner", "driver", "pilot", "sailor", "soldier", "police", "firefighter", "paramedic", "lawyer"
    ]
    
    # Add individual words as simple captions to boost vocabulary
    for word in additional_words:
        expanded_captions.append(f"beautiful {word} in nature")
        expanded_captions.append(f"amazing {word} photograph")
        expanded_captions.append(f"stunning {word} scene")
    
    # Add more complex sentence structures
    complex_templates = [
        "professional photograph showing {adj1} {obj1} during {time} with {adj2} {weather} conditions",
        "aerial view of {adj1} {location} featuring {obj1} and {obj2} under {weather} sky",
        "close-up macro photography of {adj1} {obj1} with {adj2} texture and {color} highlights",
        "landscape composition featuring {size} {obj1} in {location} during {time} with {adj1} atmosphere",
        "documentary style photograph of {adj1} {obj1} surrounded by {adj2} {obj2} in natural setting",
        "artistic black and white image of {adj1} {obj1} creating {adj2} patterns against {weather} background",
        "high resolution digital capture of {color1} {obj1} contrasting with {color2} {obj2} in {location}",
        "vintage film photography showing {adj1} {obj1} from {time} period with {adj2} aesthetic quality",
        "panoramic wide angle view of {adj1} {location} with {obj1} and {obj2} stretching to horizon",
        "intimate portrait style photograph of {adj1} {obj1} displaying {adj2} characteristics in {weather} light"
    ]
    
    # Generate more variations
    import random
    for template in complex_templates:
        for _ in range(100):  # Generate 100 variations per template
            try:
                caption = template.format(
                    adj1=random.choice(descriptors + emotions + ["professional", "artistic", "creative", "technical", "documentary"]),
                    adj2=random.choice(descriptors + ["detailed", "textured", "patterned", "structured", "organic"]),
                    obj1=random.choice(simple_objects + additional_words[:50]),
                    obj2=random.choice(simple_objects + additional_words[50:100]),
                    color=random.choice(colors),
                    color1=random.choice(colors),
                    color2=random.choice(colors),
                    size=random.choice(sizes),
                    weather=random.choice(weather),
                    time=random.choice(times),
                    location=random.choice(locations)
                )
                expanded_captions.append(caption)
            except (KeyError, IndexError):
                continue
    
    # Remove duplicates and clean up
    expanded_captions = list(set(expanded_captions))
    
    # Add start and end tokens to all captions
    processed_captions = [f"<start> {caption.lower()} <end>" for caption in expanded_captions]
    
    print(f"Generated {len(processed_captions)} unique captions")
    return processed_captions

def create_large_vocabulary_tokenizer():
    """Create tokenizer with 5000+ vocabulary"""
    print("Creating comprehensive vocabulary with 5000+ words...")
    
    # Generate comprehensive captions
    captions = create_comprehensive_captions()
    
    # Create tokenizer with large vocabulary
    tokenizer = Tokenizer(
        num_words=6000,  # Allow up to 6000 words
        oov_token='<unk>',
        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
    )
    
    # Fit on comprehensive text
    tokenizer.fit_on_texts(captions)
    
    # Ensure special tokens are properly indexed
    word_index = tokenizer.word_index
    if '<start>' not in word_index:
        word_index = {'<start>': 1, '<end>': 2, '<unk>': 3, **word_index}
    
    tokenizer.word_index = word_index
    
    vocab_size = len(word_index)
    print(f"Final vocabulary size: {vocab_size} words")
    
    # Save tokenizer
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Create and save model with large vocabulary
    max_length = 40  # Increased for more detailed captions
    model = create_captioning_model(min(vocab_size + 1, 6000), max_length)
    model.save_weights('saved_models/captioning_model.weights.h5')
    
    print("Large vocabulary tokenizer and model created successfully!")
    print(f"Vocabulary includes words like:")
    
    # Show sample vocabulary
    sample_words = list(word_index.keys())[:50]
    print(f"Sample words: {sample_words}")
    
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = create_large_vocabulary_tokenizer()