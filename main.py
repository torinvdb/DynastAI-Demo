import random
import csv
import os
import time
import sys
import base64
from datetime import datetime
import requests
from dotenv import load_dotenv
from io import BytesIO

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# External packages for enhanced UI
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from simple_term_menu import TerminalMenu

console = Console()

# Load environment variables from .env file
load_dotenv()

# Add OpenRouter API call function at top-level (after imports)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Set this in your .env file

def call_openrouter(prompt, model="google/gemini-2.5-flash-preview"):
    """Send a prompt to OpenRouter and return the response."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0.7
    }
    # Debug: Show the prompt being sent
    print("\n[DEBUG] LLM Prompt:\n", prompt)
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def ensure_pil_installed():
    """Ensure the PIL (Pillow) package is installed for image handling"""
    global PIL_AVAILABLE
    if not PIL_AVAILABLE:
        console.print("The Pillow package is required to display character images.", style="yellow")
        console.print("Attempting to install Pillow...", style="yellow")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "Pillow"])
            console.print("Pillow successfully installed!", style="green")
            # Now try to import it
            try:
                global Image, ImageOps
                from PIL import Image, ImageOps
                PIL_AVAILABLE = True
                console.print("Character images will be displayed.", style="green")
            except ImportError:
                console.print("Could not import PIL after installation. Character images will not be displayed.", style="red")
        except Exception as e:
            console.print(f"Failed to install Pillow: {e}", style="red")
            console.print("Character images will not be displayed.", style="red")
    return PIL_AVAILABLE

class KingdomGame:
    def __init__(self, csv_file="cards/cards.csv"):
        # Initialize game state
        self.power = 50      # Royal authority
        self.stability = 50  # Population happiness
        self.piety = 50      # Religious influence 
        self.wealth = 50     # Kingdom finances
        self.year = 1        # Current reign year
        self.king_name = ""
        self.dynasty = ""
        self.cards = []
        self.special_events = []
        self.history = []    # Track which cards have been played recently
        self.event_history = []  # Track triggered events
        self.traits = []     # Royal traits that affect gameplay
        self.items = []      # Special items the king can acquire
        self.advisors = {}   # Key advisors and their loyalty
        self.achievements = set()  # Tracks special accomplishments
        self.ai_play = False  # Track if AI is playing
        
        # Create directories if they don't exist
        os.makedirs("cards", exist_ok=True)
        os.makedirs("saves", exist_ok=True)
        
        # Load cards from CSV
        self.load_cards(csv_file)
        self.load_special_events()
        
    def load_cards(self, csv_file):
        """Load regular scenario cards from CSV"""
        # List of approved character roles
        approved_characters = [
            "Royal Advisor", "Royal Seneschal", "Master of Coin", "Sheriff", 
            "Master Shipwright", "Master of Laws", "Royal Jester", 
            "Master Blacksmith", "Master of the Hunt", "Court Alchemist",
            "Master of Ceremonies", "Royal Diplomat", "Royal Falconer",
            "Royal Librarian", "Court Physician", "Court Astronomer",
            "Court Philosopher", "Court Scribe", "Master of Arms",
            "Master of Spies", "Master of the Harvest", "Guild Merchant",
            "Master Builder", "Abbot", "Cardinal", "Grand Inquisitor",
            "Plague Doctor", "Princess", "Prince", "Queen",
            "Noble Lord", "Peasant", "Peasant Bailiff", "Captain of the Guard",
            "General", "Fleet Admiral", "Foreign Ambassador", "Court Jester",
            "Master of the Stables", "Master of the Kennels", "Chivalric Chapter Master"
        ]
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    try:
                        # Map CSV columns to internal stat names
                        left_effects = {
                            "piety": int(row["Left_Piety"]),
                            "stability": int(row["Left_Stability"]),
                            "power": int(row["Left_Power"]),
                            "wealth": int(row["Left_Wealth"])
                        }
                        right_effects = {
                            "piety": int(row["Right_Piety"]),
                            "stability": int(row["Right_Stability"]),
                            "power": int(row["Right_Power"]),
                            "wealth": int(row["Right_Wealth"])
                        }
                        # Only add cards with approved character roles
                        character = row.get("Character", "")
                        if character in approved_characters:
                            card = {
                                "id": row["ID"],
                                "text": row["Prompt"],
                                "left": {"text": row["Left_Choice"], "effects": left_effects},
                                "right": {"text": row["Right_Choice"], "effects": right_effects},
                                "category": row.get("Character", "general"),
                                "repeatable": row.get("Repeatable", "yes").lower() == "yes"
                            }
                            self.cards.append(card)
                    except (ValueError, KeyError) as e:
                        console.print(f"Error processing card {row.get('ID', 'unknown')}: {e}", style="red")
                        continue  # Skip this card and continue with the next one
                console.print(f"Loaded {len(self.cards)} scenario cards.", style="green")
                
            # Try loading additional card files if initial load was successful
            if self.cards:
                additional_files = []
                # Check if we loaded cards-5.csv
                if csv_file == "cards/cards-5.csv":
                    pass  # Already loaded the main file with approved characters
                else:
                    additional_files.append("cards/cards-5.csv")
                
                # Load additional card files if they exist
                for add_file in additional_files:
                    if os.path.exists(add_file):
                        try:
                            with open(add_file, 'r', encoding='utf-8') as file:
                                reader = csv.DictReader(file)
                                card_count = len(self.cards)
                                for row in reader:
                                    try:
                                        # Map CSV columns to internal stat names
                                        left_effects = {
                                            "piety": int(row["Left_Piety"]),
                                            "stability": int(row["Left_Stability"]),
                                            "power": int(row["Left_Power"]),
                                            "wealth": int(row["Left_Wealth"])
                                        }
                                        right_effects = {
                                            "piety": int(row["Right_Piety"]),
                                            "stability": int(row["Right_Stability"]),
                                            "power": int(row["Right_Power"]),
                                            "wealth": int(row["Right_Wealth"])
                                        }
                                        # Only add cards with approved character roles
                                        character = row.get("Character", "")
                                        if character in approved_characters:
                                            card = {
                                                "id": row["ID"],
                                                "text": row["Prompt"],
                                                "left": {"text": row["Left_Choice"], "effects": left_effects},
                                                "right": {"text": row["Right_Choice"], "effects": right_effects},
                                                "category": row.get("Character", "general"),
                                                "repeatable": row.get("Repeatable", "yes").lower() == "yes"
                                            }
                                            self.cards.append(card)
                                    except (ValueError, KeyError) as e:
                                        continue  # Skip this card and continue with the next one
                                console.print(f"Loaded {len(self.cards) - card_count} additional cards from {add_file}.", style="green")
                        except Exception as e:
                            console.print(f"Error loading additional cards from {add_file}: {e}", style="yellow")
            
        except FileNotFoundError:
            console.print(f"Card file {csv_file} not found. Creating sample cards instead.", style="yellow")
            self.create_sample_cards()
            
    def load_special_events(self):
        """Load special events that trigger based on conditions"""
        # These are special narrative events that can trigger based on specific conditions
        self.special_events = [
            {
                "id": "royal_wedding",
                "Character": "Royal Diplomat",
                "trigger": lambda self: self.year > 3 and "royal_wedding" not in self.event_history,
                "text": "The Royal Diplomat approaches with news: A neighboring kingdom proposes a royal marriage alliance.",
                "left": {
                    "text": "Decline the offer",
                    "effects": {"power": -5, "stability": 0, "piety": 0, "wealth": 0},
                    "outcome": "The neighboring kingdom is offended. Relations deteriorate."
                },
                "right": {
                    "text": "Accept the alliance", 
                    "effects": {"power": 10, "stability": 5, "piety": 5, "wealth": -10},
                    "outcome": "The royal wedding is magnificent. Your kingdom gains a powerful ally.",
                    "add_trait": "Allied"
                }
            },
            {
                "id": "plague",
                "Character": "Court Physician",
                "trigger": lambda self: self.year > 5 and random.random() < 0.2 and "plague" not in self.event_history,
                "text": "The Court Physician appears pale: A terrible plague has broken out in the outer provinces!",
                "left": {
                    "text": "Quarantine affected areas",
                    "effects": {"power": 0, "stability": -15, "piety": 10, "wealth": -10},
                    "outcome": "The plague is contained, but at a terrible cost to the isolated communities."
                },
                "right": {
                    "text": "Send doctors and aid",
                    "effects": {"power": 0, "stability": 10, "piety": 5, "wealth": -20},
                    "outcome": "Your swift response saves many lives, though the treasury is drained."
                }
            },
            {
                "id": "golden_age",
                "Character": "Royal Advisor",
                "trigger": lambda self: all(getattr(self, stat) > 70 for stat in ["power", "stability", "piety", "wealth"]),
                "text": "The Royal Advisor bows deeply: Your wise rule has ushered in a Golden Age of prosperity and happiness!",
                "left": {
                    "text": "Commemorate with festivals",
                    "effects": {"power": 5, "stability": 10, "piety": 5, "wealth": -10},
                    "outcome": "The people celebrate your magnificent reign with joyous festivals.",
                    "add_achievement": "Golden Age"
                },
                "right": {
                    "text": "Build monuments",
                    "effects": {"power": 10, "stability": -5, "piety": 5, "wealth": -15},
                    "outcome": "Grand monuments are constructed to immortalize your dynasty.",
                    "add_achievement": "Golden Age"
                }
            },
            {
                "id": "coronation",
                "Character": "Royal Seneschal",
                "trigger": lambda self: self.year == 1,
                "text": f"The Royal Seneschal approaches: The crown sits heavy upon your brow. How will you mark the beginning of your reign?",
                "left": {
                    "text": "Generous coronation gifts",
                    "effects": {"power": 0, "stability": 15, "piety": 5, "wealth": -15},
                    "outcome": "The people cheer as gold coins are distributed throughout the kingdom."
                },
                "right": {
                    "text": "Military parade",
                    "effects": {"power": 15, "stability": -5, "piety": 0, "wealth": -5},
                    "outcome": "Your strength impresses allies and enemies alike."
                }
            }
        ]
            
    def create_sample_cards(self):
        """Create sample cards if CSV file is not found"""
        self.cards = [
            {"id": "1", "text": "The Royal Advisor suggests lowering taxes to appease the populace.", 
             "left": {"text": "Refuse", "effects": {"power": 10, "stability": -15, "piety": 0, "wealth": 15}},
             "right": {"text": "Accept", "effects": {"power": -5, "stability": 20, "piety": 0, "wealth": -15}},
             "category": "general", "repeatable": True},
            
            {"id": "2", "text": "The Cardinal asks for funds to build a new cathedral.", 
             "left": {"text": "Deny funding", "effects": {"power": 0, "stability": 5, "piety": -15, "wealth": 10}},
             "right": {"text": "Fund the cathedral", "effects": {"power": 5, "stability": -5, "piety": 20, "wealth": -15}},
             "category": "piety", "repeatable": True},
             
            {"id": "3", "text": "The General requests more troops for the army.", 
             "left": {"text": "Maintain current size", "effects": {"power": -10, "stability": 5, "piety": 0, "wealth": 10}},
             "right": {"text": "Expand the army", "effects": {"power": 15, "stability": -10, "piety": 0, "wealth": -15}},
             "category": "power", "repeatable": True},
             
            {"id": "4", "text": "The Royal Diplomat reports a nearby kingdom offers a trade agreement.", 
             "left": {"text": "Refuse", "effects": {"power": 5, "stability": -5, "piety": 0, "wealth": -10}},
             "right": {"text": "Accept", "effects": {"power": -5, "stability": 10, "piety": 0, "wealth": 15}},
             "category": "wealth", "repeatable": False},
             
            {"id": "5", "text": "The Court Astronomer reports strange lights appearing in the night sky.", 
             "left": {"text": "It's an omen", "effects": {"power": 0, "stability": -10, "piety": 15, "wealth": 0}},
             "right": {"text": "It's just stars", "effects": {"power": 0, "stability": 5, "piety": -10, "wealth": 0}},
             "category": "event", "repeatable": False},
        ]
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
            
    def display_title(self):
        """Display the game title with rich formatting"""
        title = """
        ▓█████▄▓██   ██▓ ███▄    █  ▄▄▄        ██████ ▄▄▄█████▓ ▄▄▄       ██▓
        ▒██▀ ██▌▒██  ██▒ ██ ▀█   █ ▒████▄    ▒██    ▒ ▓  ██▒ ▓▒▒████▄    ▓██▒
        ░██   █▌ ▒██ ██░▓██  ▀█ ██▒▒██  ▀█▄  ░ ▓██▄   ▒ ▓██░ ▒░▒██  ▀█▄  ▒██▒
        ░▓█▄   ▌ ░ ▐██▓░▓██▒  ▐▌██▒░██▄▄▄▄██   ▒   ██▒░ ▓██▓ ░ ░██▄▄▄▄██ ░██░
        ░▒████▓  ░ ██▒▓░▒██░   ▓██░ ▓█   ▓██▒▒██████▒▒  ▒██▒ ░  ▓█   ▓██▒░██░
        ▒▒▓  ▒   ██▒▒▒ ░ ▒░   ▒ ▒  ▒▒   ▓▒█░▒ ▒▓▒ ▒ ░  ▒ ░░    ▒▒   ▓▒█░░▓  
        ░ ▒  ▒ ▓██ ░▒░ ░ ░░   ░ ▒░  ▒   ▒▒ ░░ ░▒  ░ ░    ░      ▒   ▒▒ ░ ▒ ░
        ░ ░  ░ ▒ ▒ ░░     ░   ░ ░   ░   ▒   ░  ░  ░    ░        ░   ▒    ▒ ░
        ░    ░ ░              ░       ░  ░      ░                 ░  ░ ░  
        ░      ░ ░                                                          """
        console.print(Panel(title, style="yellow", border_style="red"))
        console.print("  * Rule wisely or suffer the consequences *\n", style="bold red")
        
    def display_status(self):
        """Display the current game state with rich progress bars"""
        self.clear_screen()
        
        # Create title with year and traits
        title = f"Year {self.year} of {self.king_name}'s reign"
        if self.traits:
            title += f" | Traits: {', '.join(self.traits)}"
        
        # Display items if any
        items_text = f"Items: {', '.join(self.items)}" if self.items else ""
        
        # Create a table for the stats
        table = Table(show_header=False, box=None)
        table.add_column("Stat", style="cyan")
        table.add_column("Bar")
        table.add_column("Value", style="white")
        
        # Add stat rows with progress bars
        stat_colors = {"Power": "red", "Stability": "green", "Piety": "yellow", "Wealth": "blue"}
        for stat_name, internal_name in [("Power", "power"), ("Stability", "stability"), 
                                        ("Piety", "piety"), ("Wealth", "wealth")]:
            value = getattr(self, internal_name)
            
            # Create a custom progress bar with consistent text color
            progress = Progress(
                TextColumn(""),
                BarColumn(bar_width=40, style=stat_colors[stat_name]),
                TextColumn("[white]{task.completed}/100[/white] [white]{task.percentage:>3.0f}%[/white]"),
                expand=False
            )
            progress.add_task(f"", total=100, completed=value)
            
            # Add indicator for dangerous levels
            indicator = ""
            if value < 20:
                indicator = "[bold red]!"
            elif value > 80:
                indicator = "[bold red]!"
                
            table.add_row(f"{stat_name}:", progress, f"{indicator}")
        
        # Create advisor table if any
        advisors_table = None
        if self.advisors:
            advisors_table = Table(title="Advisors", show_header=False, box=None)
            advisors_table.add_column("Name", style="blue")
            advisors_table.add_column("Loyalty")
            
            for advisor, loyalty in self.advisors.items():
                # Color based on loyalty
                color = "red" if loyalty < 30 else "yellow" if loyalty < 70 else "green"
                loyalty_bar = "█" * (loyalty // 10)
                advisors_table.add_row(advisor, f"[{color}]{loyalty_bar}[/{color}] {loyalty}")
        
        # Prepare the renderables for the panel
        renderables = [table]
        if advisors_table:
            renderables.append(advisors_table)
        
        # Display everything in a panel using Group to combine renderables
        console.print(Panel(
            Group(
                Text(title, style="bold"),
                Text(items_text, style="italic") if items_text else "",
                *renderables
            ),
            title="Kingdom Status",
            border_style="cyan"
        ))
        
    def get_ruler_epithet(self):
        """Generate an epithet for the ruler based on their stats and achievements"""
        # Base epithet on highest stat
        stats = {
            "power": self.power,
            "stability": self.stability,
            "piety": self.piety,
            "wealth": self.wealth
        }
        
        # Get highest and lowest stats
        highest_stat = max(stats, key=stats.get)
        lowest_stat = min(stats, key=stats.get)
        
        # Choose epithet based on stats
        epithets = {
            "power": ["the Iron-Fisted", "the Authoritarian", "the Mighty", "the Conqueror"],
            "stability": ["the Beloved", "the People's Champion", "the Kind", "the Generous"],
            "piety": ["the Pious", "the Holy", "the Devout", "the Blessed"],
            "wealth": ["the Wealthy", "the Rich", "the Prosperous", "the Merchant King"]
        }
        
        # Choose negative epithets based on lowest stat
        negative_epithets = {
            "power": ["the Weak", "the Spineless"],
            "stability": ["the Cruel", "the Tyrant"],
            "piety": ["the Heretic", "the Godless"],
            "wealth": ["the Wasteful", "the Bankrupt"]
        }
        
        # Choose epithet based on highest and lowest stats and their values
        if stats[highest_stat] > 75:
            epithet = random.choice(epithets[highest_stat])
        elif stats[lowest_stat] < 25:
            epithet = random.choice(negative_epithets[lowest_stat])
        elif self.year > 40:
            epithet = "the Long-Reigning"
        elif "Golden Age" in self.achievements:
            epithet = "the Magnificent"
        elif "Wise" in self.traits:
            epithet = "the Wise"
        else:
            # Default epithets based on reign length
            if self.year < 10:
                epithet = "the Brief"
            elif self.year < 20:
                epithet = "the Adequate"
            else:
                epithet = "the Steady"
                
        return epithet
        
    def get_detailed_ending(self, end_condition_message):
        """Generate a detailed ending based on what caused the game to end"""
        
        # Create base context for the ending
        context = {
            "power": "high" if self.power > 75 else "low" if self.power < 25 else "moderate",
            "stability": "high" if self.stability > 75 else "low" if self.stability < 25 else "moderate",
            "piety": "high" if self.piety > 75 else "low" if self.piety < 25 else "moderate",
            "wealth": "high" if self.wealth > 75 else "low" if self.wealth < 25 else "moderate",
            "years": self.year,
            "traits": ", ".join(self.traits) if self.traits else "unremarkable",
        }
        
        # Detailed endings based on cause of game over
        detailed_endings = {
            "power_low": f"Years of concessions and weak leadership eroded your authority. The nobles, seeing your weakness, formed a coalition against you. After a brief struggle, you were deposed and exiled, remembered as a ruler who couldn't maintain the respect of the nobility.",
            
            "power_high": f"Your iron-fisted rule and consolidation of power bred resentment among the nobility. As your authority grew unchecked, many feared for their own positions. A conspiracy formed in the shadows, and despite your vigilance, an assassin's blade found its mark. You died as you ruled - alone and feared.",
            
            "stability_low": f"The cries of the hungry and oppressed grew too loud to ignore. Years of neglect and harsh policies turned the populace against you. What began as isolated protests quickly spread across the kingdom. The uprising was swift and merciless, with angry mobs storming the palace. Your reign ended at the hands of those you failed to serve.",
            
            "stability_high": f"The common folk adored you for your generosity and fairness. However, your popularity threatened the traditional power structure. As people began calling for democratic reforms and greater representation, the nobles and church became alarmed. They orchestrated your removal, claiming the kingdom needed 'proper governance, not popularity.' The republic that followed bore your name, though you did not live to see it flourish.",
            
            "piety_low": f"Your dismissal of religious traditions and constant conflicts with church authorities were deemed heretical. The Grand Inquisitor publicly denounced you, turning religious sentiment against the crown. Priests preached against you from every pulpit until the faithful rose up in a holy crusade. Declared a heretic, you faced the ultimate punishment for challenging divine authority.",
            
            "piety_high": f"You allowed religious authorities too much influence, and the church's power grew unchecked. Gradually, religious law superseded royal edicts, and church officials began overruling your decisions. Eventually, the Archbishop declared divine right to rule, and with popular support, established a theocracy. You were permitted to retain your title in name only - a figurehead in a kingdom ruled by the cloth.",
            
            "wealth_low": f"Years of extravagance and financial mismanagement emptied the royal coffers. Unable to pay the army or maintain the kingdom's infrastructure, your rule collapsed under mounting debts. Foreign creditors seized royal assets, while unpaid servants and soldiers abandoned their posts. With nothing left to rule, you were quietly removed from the throne, your name becoming synonymous with fiscal irresponsibility.",
            
            "wealth_high": f"Your kingdom's legendary wealth attracted unwanted attention. Neighboring rulers looked upon your treasuries with envy, and despite your diplomatic efforts, greed won out. A coalition of foreign powers, using your hoarding of wealth as justification, invaded with overwhelming force. Your vast riches funded your enemies' armies, and your kingdom was divided among the victors.",
            
            "old_age": f"After {self.year} years of rule, age finally caught up with you. Your legacy secured, you passed peacefully in your sleep, surrounded by generations of family. The kingdom mourned for forty days, and your achievements were recorded in detail by royal historians. Few monarchs are fortunate enough to meet such a natural end, a testament to your balanced approach to leadership."
        }
        
        # Determine which detailed ending to use based on the end condition message
        if "authority" in end_condition_message:
            detail = detailed_endings["power_low"]
        elif "tyrant" in end_condition_message:
            detail = detailed_endings["power_high"]
        elif "revolt" in end_condition_message:
            detail = detailed_endings["stability_low"]
        elif "republic" in end_condition_message:
            detail = detailed_endings["stability_high"]
        elif "heretic" in end_condition_message:
            detail = detailed_endings["piety_low"]
        elif "too powerful" in end_condition_message and "church" in end_condition_message:
            detail = detailed_endings["piety_high"]
        elif "bankrupt" in end_condition_message:
            detail = detailed_endings["wealth_low"]
        elif "wealthy" in end_condition_message:
            detail = detailed_endings["wealth_high"]
        elif "old age" in end_condition_message:
            detail = detailed_endings["old_age"]
        else:
            # Default generic ending if none of the specific conditions are met
            detail = f"After ruling for {self.year} years, your reign came to an end. {end_condition_message}"
        
        return detail
    
    def apply_effects(self, effects, trait_modifiers=True):
        """Apply effects to game stats and ensure they stay within bounds"""
        changes = {}
        
        # Apply trait modifiers to effects
        modified_effects = effects.copy()
        if trait_modifiers:
            if "Wise" in self.traits:
                # Wise rulers get better outcomes
                for stat in modified_effects:
                    if modified_effects[stat] > 0:
                        modified_effects[stat] += 2
                    elif modified_effects[stat] < 0:
                        modified_effects[stat] += 1
                        
            if "Cruel" in self.traits:
                # Cruel rulers have worse stability outcomes
                if "stability" in modified_effects and modified_effects["stability"] > 0:
                    modified_effects["stability"] -= 5
                
            if "Pious" in self.traits:
                # Pious rulers get better piety outcomes
                if "piety" in modified_effects:
                    modified_effects["piety"] += 3
                    
            if "Merchant" in self.traits:
                # Merchant rulers get better wealth outcomes
                if "wealth" in modified_effects:
                    modified_effects["wealth"] += 3
                    
        # Apply the modified effects
        for stat, change in modified_effects.items():
            old_value = getattr(self, stat)
            new_value = max(0, min(100, old_value + change))
            setattr(self, stat, new_value)
            changes[stat] = new_value - old_value
            
        return changes
    
    def check_special_events(self):
        """Check if any special events should trigger"""
        for event in self.special_events:
            if event["trigger"](self) and event["id"] not in self.event_history:
                return event
        return None
        
    def get_random_card(self):
        """Get a random card, avoiding recently played cards"""
        # First check for special events
        special_event = self.check_special_events()
        if special_event:
            self.event_history.append(special_event["id"])
            return special_event
            
        # Filter out non-repeatable cards that have been played
        available_cards = [card for card in self.cards 
                          if card["repeatable"] or card["id"] not in self.history]
        
        # Also filter out recently played cards
        recent_cards = self.history[-5:] if self.history else []
        available_cards = [card for card in available_cards if card["id"] not in recent_cards]
        
        # If we've exhausted unique cards, reset history
        if not available_cards:
            self.history = []
            available_cards = self.cards
            
        card = random.choice(available_cards)
        self.history.append(card["id"])
        return card
    
    def save_game(self):
        """Save the current game state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saves/{self.king_name.lower().replace(' ', '_')}_{timestamp}.save"
        
        save_data = {
            "king_name": self.king_name,
            "dynasty": self.dynasty,
            "year": self.year,
            "power": self.power,
            "stability": self.stability,
            "piety": self.piety,
            "wealth": self.wealth,
            "traits": self.traits,
            "items": self.items,
            "advisors": self.advisors,
            "history": self.history,
            "event_history": self.event_history,
            "achievements": list(self.achievements)
        }
        
        try:
            with open(filename, 'w') as file:
                for key, value in save_data.items():
                    file.write(f"{key}:{value}\n")
            console.print(f"\nGame saved as {filename}", style="green")
            time.sleep(1.5)
        except Exception as e:
            console.print(f"\nError saving game: {e}", style="red")
            time.sleep(1.5)
    
    def handle_special_card_outcome(self, card, choice):
        """Handle special outcomes from event cards"""
        outcome_key = "outcome" if isinstance(card, dict) else f"{choice}_outcome"
        trait_key = f"add_trait" if choice == "right" else "add_left_trait"
        achieve_key = "add_achievement"
        item_key = "add_item"
        advisor_key = "add_advisor"
        
        # Display outcome text if available
        if outcome_key in card:
            console.print(f"\n{card[outcome_key]}", style="cyan")
            
        # Add trait if specified
        if trait_key in card and card[trait_key] not in self.traits:
            self.traits.append(card[trait_key])
            console.print(f"\nYou gained the trait: {card[trait_key]}", style="magenta")
            
        # Add achievement if specified
        if achieve_key in card and card[achieve_key] not in self.achievements:
            self.achievements.add(card[achieve_key])
            console.print(f"\nAchievement unlocked: {card[achieve_key]}", style="yellow")
            
        # Add item if specified
        if item_key in card and card[item_key] not in self.items:
            self.items.append(card[item_key])
            console.print(f"\nYou acquired: {card[item_key]}", style="green")
            
        # Add advisor if specified
        # if advisor_key in card and card[advisor_key] not in self.advisors:
        #     self.advisors[card[advisor_key]] = 50  # Default loyalty
        #     console.print(f"\n{card[advisor_key]} has joined your court!", style="blue")
        
    def get_character_image_path(self, character):
        """Get the path to the character image file"""
        # Convert character name to lowercase and replace spaces with hyphens
        character_filename = character.lower().replace(" ", "-").replace("_", "-")
        filename = f"img/{character_filename}.png"
        if os.path.exists(filename):
            return filename
        return None
    
    def play(self):
        """Main game loop"""
        self.clear_screen()
        self.display_title()
        
        # Character creation
        self.king_name = console.input("\nEnter your royal name: ")
        if not self.king_name:
            self.king_name = "Monarch"
            
        self.dynasty = console.input("Enter your dynasty name: ")
        if not self.dynasty:
            self.dynasty = "House of " + self.king_name
            
        # Choose starting trait with terminal menu
        console.print("\nChoose your royal trait:", style="cyan")
        traits = ["Wise", "Pious", "Merchant", "Warrior"]
        trait_menu = TerminalMenu(traits, title="Select one trait:")
        trait_index = trait_menu.show()
        if trait_index is not None:  # In case of KeyboardInterrupt
            self.traits.append(traits[trait_index])
        else:
            self.traits.append("Wise")  # Default if user cancels
        
        # Ask if user or AI should play
        console.print("\nWho will play this reign?", style="cyan")
        player_options = ["Human (You)", "AI (OpenRouter LLM)"]
        player_menu = TerminalMenu(player_options, title="Choose player:")
        player_choice = player_menu.show()
        self.ai_play = (player_choice == 1)
        
        console.print(f"\nAll hail {self.king_name} of the {self.dynasty}, long may you reign!", style="yellow")
        console.print("\nYour goal is to balance the four aspects of your kingdom:", style="cyan")
        console.print("- Power (royal authority)")
        console.print("- Stability (population happiness)")
        console.print("- Piety (religious influence)")
        console.print("- Wealth (royal treasury)")
        console.print("\nIf any value reaches 0 or 100, your reign will end.", style="red")
        console.print("Make your choices wisely...\n", style="yellow")
        
        console.input("Press Enter to begin your reign...")
        
        # Main game loop
        while True:
            self.display_status()
            
            game_over_message = self.check_game_over()
            if game_over_message:
                epithet = self.get_ruler_epithet()
                detailed_ending = self.get_detailed_ending(game_over_message)
                
                # Display game over message and epithet
                console.print(f"\n{game_over_message}", style="red")
                console.print(f"\n{self.king_name} {epithet} ruled for {self.year} years.", style="yellow")
                
                # Display detailed ending narrative
                console.print(Panel(detailed_ending, title="The End of Your Reign", border_style="red"))
                
                # Display historical legacy
                legacy_message = f"{self.king_name} {epithet} will be remembered as "
                
                # Determine legacy based on stats
                if self.year < 5:
                    legacy_message += "barely a footnote in the kingdom's history."
                elif "Golden Age" in self.achievements:
                    legacy_message += "one of the greatest rulers the kingdom has ever known."
                elif self.power > 75 and self.wealth > 75:
                    legacy_message += "a powerful and wealthy monarch who expanded the kingdom's influence."
                elif self.stability > 75 and self.piety > 75:
                    legacy_message += "a beloved leader who unified the people and church in harmonious prosperity."
                elif self.stability > 75:
                    legacy_message += "a ruler of the people, cherished in common memory for generations to come."
                elif self.piety > 75:
                    legacy_message += "a faithful defender of the faith, whose piety set an example for all."
                elif self.power > 75:
                    legacy_message += "an iron-fisted ruler whose authority was never questioned."
                elif self.wealth > 75:
                    legacy_message += "a savvy economic mind who filled the kingdom's coffers with gold."
                elif all(25 <= getattr(self, stat) <= 75 for stat in ["power", "stability", "piety", "wealth"]):
                    legacy_message += "a balanced ruler who maintained stability in turbulent times."
                else:
                    legacy_message += "a ruler of mixed fortunes, with as many failures as successes."
                
                console.print(Panel(legacy_message, border_style="yellow"))
                
                # Display achievements
                if self.achievements:
                    console.print("\nAchievements unlocked:", style="cyan")
                    for achievement in self.achievements:
                        console.print(f"- {achievement}", style="yellow")
                break
                
            # Present a card
            card = self.get_random_card()
            
            # Determine if it's a special event
            is_special = "trigger" in card
            
            # Show character image or name
            character_name = None
            if not is_special and "category" in card:
                character_name = card["category"]
            elif not is_special and "Character" in card:
                character_name = card["Character"]
            elif is_special and "Character" in card:
                character_name = card["Character"]
                
            if character_name:
                image_path = self.get_character_image_path(character_name)
                if image_path and PIL_AVAILABLE:
                    try:
                        # Display the character name with styling
                        console.print(f"[bold cyan]{character_name}[/bold cyan]")
                        
                        # Load the image using Pillow
                        img = Image.open(image_path)
                        
                        # Terminal-friendly image conversion
                        # Resize image to fit in terminal (maintain aspect ratio)
                        width, height = img.size
                        aspect_ratio = height / width
                        new_width = 40  # Terminal-friendly width
                        new_height = int(aspect_ratio * new_width * 0.5)  # Adjust for terminal character height/width ratio
                        
                        # Handle different versions of Pillow
                        try:
                            # For newer Pillow versions
                            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        except AttributeError:
                            # For older Pillow versions
                            img = img.resize((new_width, new_height), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
                        
                        # Convert to grayscale for simplicity
                        img = img.convert('L')
                        
                        # Use ASCII characters to represent different brightness levels
                        # More detailed character set for better representation
                        chars = '@%#*+=-:. '[::-1]  # Reversed for better contrast
                        ascii_art = []
                        for y in range(new_height):
                            line = ""
                            for x in range(new_width):
                                # Get pixel brightness (0-255)
                                brightness = img.getpixel((x, y))
                                # Map brightness to a character
                                char_idx = min(int(brightness * len(chars) / 256), len(chars) - 1)
                                line += chars[char_idx]
                            ascii_art.append(line)
                            
                        # Add spacing above and below the ASCII art for better visibility
                        console.print("")
                        for line in ascii_art:
                            console.print(line)
                        console.print("")
                    except Exception as e:
                        # If image display fails, just show the character name
                        console.print(f"[bold cyan]{character_name}[/bold cyan]")
                        console.print(f"[dim](Image display error: {str(e)})[/dim]")
                else:
                    # If PIL is not available or image not found, just show the character name
                    console.print(f"[bold cyan]{character_name}[/bold cyan]")

            if is_special:
                console.print("\n--- Special Event ---", style="magenta")
                console.print(Panel(card['text'], style="cyan"))
            else:
                console.print(Panel(card['text'], style="white"))
                
            # Create menu options
            options = [
                f"1. {card['left']['text']}", 
                f"2. {card['right']['text']}",
                "S. Save Game",
                "Q. Quit Game"
            ]
            
            # Show menu and get choice
            if self.ai_play:
                # Format the prompt for the LLM
                prompt = (
                    f"You are the monarch. Here is the scenario:\n"
                    f"{card['text']}\n"
                    f"1. {card['left']['text']}\n"
                    f"2. {card['right']['text']}\n"
                    "Reply ONLY with 1 or 2 to make your choice."
                )
                print(f"[DEBUG] LLM Prompt: {prompt}")  # Print the prompt
                llm_response = call_openrouter(prompt)
                # Debug: Show the LLM's raw response
                print(f"[DEBUG] LLM Response: {llm_response}")
                if "1" in llm_response:
                    choice_index = 0
                elif "2" in llm_response:
                    choice_index = 1
                else:
                    # Default to left if unclear
                    choice_index = 0
            else:
                menu = TerminalMenu(options, title="What will you do?")
                choice_index = menu.show()
            
            # Handle choice
            if choice_index == 2:  # Save Game
                self.save_game()
                continue
            elif choice_index == 3:  # Quit Game
                confirm_menu = TerminalMenu(["Yes", "No"], title="Are you sure you want to quit?")
                confirm = confirm_menu.show()
                if confirm == 0:  # Yes
                    return False
                continue
            elif choice_index == 0:  # Left choice
                effects = card['left']['effects'] if 'left' in card and 'effects' in card['left'] else {}
                choice_text = card['left']['text']
                choice_key = "left"
            else:  # Right choice
                effects = card['right']['effects'] if 'right' in card and 'effects' in card['right'] else {}
                choice_text = card['right']['text']
                choice_key = "right"
            
            # Apply effects and get actual changes
            console.print(f"\nYou chose: {choice_text}", style="cyan")
            changes = self.apply_effects(effects)
            
            # Handle any special outcomes
            if is_special or "outcome" in card.get(choice_key, {}):
                self.handle_special_card_outcome(card, choice_key)
            
            # Show how stats changed
            change_messages = []
            for stat, change in changes.items():
                if change > 0:
                    change_messages.append(f"[green]{stat.capitalize()} +{change}[/green]")
                elif change < 0:
                    change_messages.append(f"[red]{stat.capitalize()} {change}[/red]")
            
            if change_messages:
                console.print("Effects: " + " ".join(change_messages))
            
            self.year += 1
            
            # Random advisor loyalty changes
            # for advisor in self.advisors:
            #     # Small random changes to loyalty based on decisions
            #     if random.random() < 0.3:  # 30% chance for each advisor
            #         change = random.randint(-5, 5)
            #         self.advisors[advisor] = max(0, min(100, self.advisors[advisor] + change))
            
            # Random trait acquisition (small chance)
            if random.random() < 0.05:  # 5% chance each turn
                potential_traits = ["Wise", "Cruel", "Pious", "Scholar", "Warrior", "Merchant"]
                new_traits = [t for t in potential_traits if t not in self.traits]
                if new_traits:
                    new_trait = random.choice(new_traits)
                    self.traits.append(new_trait)
                    console.print(f"\nYou've gained the trait: {new_trait}", style="magenta")
            
            # Pause briefly before the next turn
            time.sleep(1.5)
        
        # Game over
        console.print("\nGame Over", style="red")
        
        # Ask to play again
        play_again_menu = TerminalMenu(["Yes", "No"], title="Play again?")
        play_again = play_again_menu.show()
        return play_again == 0  # 0 = Yes, 1 = No

    def check_game_over(self):
        """Check if any stat has reached its minimum or maximum value"""
        # Dictionary of game over conditions and their messages
        end_conditions = {
            lambda: self.power <= 0: "You have lost all authority. The nobles have overthrown you!",
            lambda: self.power >= 100: "Your absolute power has made you a tyrant. You were assassinated!",
            lambda: self.stability <= 0: "The people are in open revolt! You have been deposed!",
            lambda: self.stability >= 100: "The people love you so much they've decided to establish a republic!",
            lambda: self.piety <= 0: "The church has declared you a heretic. You were excommunicated and executed!",
            lambda: self.piety >= 100: "The church has become too powerful and taken control of the kingdom!",
            lambda: self.wealth <= 0: "The treasury is empty. The kingdom is bankrupt and you have been deposed!",
            lambda: self.wealth >= 100: "Your kingdom is so wealthy that a rival monarch invaded to seize your riches!",
            lambda: self.year >= 80 and "Immortal" not in self.traits: f"{self.king_name} died of old age after a long and eventful reign."
        }
        
        # Check each condition
        for condition, message in end_conditions.items():
            if condition():
                return message
                
        return False

def main():
    """Main function to run the game"""
    try:
        # Try to install PIL if not available
        ensure_pil_installed()
        
        while True:
            game = KingdomGame()
            play_again = game.play()
            if not play_again:
                break
                
        console.print("\nThank you for playing Kingdom Cards!", style="green")
        
    except KeyboardInterrupt:
        console.print("\n\nGame terminated by user. Farewell!", style="yellow")
    except Exception as e:
        console.print(f"\n\nAn unexpected error occurred: {e}", style="red")
        console.print("The game has been terminated.")

if __name__ == "__main__":
    main()