import pygame
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import defaultdict

pygame.init()

# Constants
WIDTH, HEIGHT = 1024, 768
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
LAST_TRAIN_BUFFER = 30  # 30 menit setelah kereta terakhir

# Simulation parameters
SIMULATION_SPEED = 60  # 60x real time (1 second = 1 minute in simulation)
MAX_SIMULATION_TIME = 24 * 60  # 24 hours in minutes
PASSENGER_GIVE_UP_WAIT_TIME = 120  # Passengers give up after 2 hours
UPCOMING_TRAIN_CHECK_WINDOW = 120 # Check for upcoming trains within this window (minutes) for passenger generation

# Station data: (name, jarak km (kurang lebih))
STATIONS = [
    ("YK", 0),       # st Yogyakarta
    ("LPN", 3),      # Lempuyangan
    ("MGW", 9),     # Maguwo
    ("KT", 30),      # Klaten
    ("PWS", 60),     # Purwosari
    ("SLO", 64)      # Solo Balapan
]

# Jadwal Kereta: (waktu keberangkatan dari Yogya dalam menit dari 00:00, kapasitas)
TRAIN_SCHEDULE = [
    (5 * 60 + 5, 1600),     # 05:05
    (6 * 60 + 0, 1600),     # 06:00
    (7 * 60 + 5, 1600),     # 07:05
    (7 * 60 + 54, 1600),    # 07:54
    (8 * 60 + 49, 1600),    # 08:49
    (10 * 60 + 56, 1600),   # 10:56
    (12 * 60 + 7, 1600),    # 12:07
    (13 * 60 + 57, 1600),   # 13:57
    (15 * 60 + 1, 1600),    # 15:01
    (16 * 60 + 10, 1600),   # 16:10
    (17 * 60 + 35, 1600),   # 17:35
    (18 * 60 + 8, 1600),    # 18:08
    (20 * 60 + 15, 1600),   # 20:15
    (21 * 60 + 20, 1600),   # 21:20
    (22 * 60 + 35, 1600),   # 22:35
]

# Rate penumpang per jam per stasiun
PASSENGER_RATES = {}
for hour in range(24):
    PASSENGER_RATES[hour] = {}
    for station_name, _ in STATIONS:
        # Default low rate for early morning and late night (0-5, 24)
        rate = 0.75
        
        # PAGI SIBUK (6-8 AM) - Rush hour pagi hari
        if 6 <= hour <= 8:
            match station_name:
                case "YK":
                    rate = 14.0  # Yogya sebagai stasiun awal
                case "LPN":
                    rate = 4.0   # Lempuyangan cukup ramai
                case "MGW":
                    rate = 2.0   # Maguwo (airport commuters)
                case "KT":
                    rate = 5.0   # Klaten hub utama
                case "PWS":
                    rate = 0.5   # Purwosari sedang
                case "SLO":
                    rate = 0.1   # Solo sedikit (stasiun akhir)
        
        # TRANSISI PAGI (9 AM) - Sisa jam sibuk pagi
        elif hour == 9:
            match station_name:
                case "YK":
                    rate = 12.0
                case "LPN":
                    rate = 3.0
                case "MGW":
                    rate = 2.0
                case "KT":
                    rate = 4.0
                case "PWS":
                    rate = 0.5
                case "SLO":
                    rate = 0.05
        
        # SIANG HARI (10-15) - Jam kerja normal, traffic sedang
        elif 10 <= hour <= 15:
            match station_name:
                case "YK":
                    rate = 7.0   
                case "LPN":
                    rate = 1.5   
                case "MGW":
                    rate = 1.0  
                case "KT":
                    rate = 1.5  
                case "PWS":
                    rate = 0.8   
                case "SLO":
                    rate = 0.1
        
        # SORE SIBUK (16-17) - Peak hour sore hari
        elif 16 <= hour <= 18:
            match station_name:
                case "YK":
                    rate = 18.0  # Peak tertinggi untuk jam pulang kerja
                case "LPN":
                    rate = 6.0   
                case "MGW":
                    rate = 4.0   
                case "KT":
                    rate = 6.0   # Banyak yang pulang kerja
                case "PWS":
                    rate = 3.0   
                case "SLO":
                    rate = 0.5
        
        # SORE RAMAI (18-21) - Jam sibuk sore lanjutan
        elif 19<= hour <= 21:
            match station_name:
                case "YK":
                    rate = 14.0  # Masih tinggi tapi mulai menurun
                case "LPN":
                    rate = 4.0   
                case "MGW":
                    rate = 3.0   
                case "KT":
                    rate = 4.0   
                case "PWS":
                    rate = 2.5   
                case "SLO":
                    rate = 0.8
        
        # MALAM (22-23) - Masih ada tapi menurun
        elif 22 <= hour <= 23:
            match station_name:
                case "YK":
                    rate = 4.0   # Masih ada yang pulang malam
                case "LPN":
                    rate = 1.2
                case "MGW":
                    rate = 1.0
                case "KT":
                    rate = 2.0   # Masih agak tinggi
                case "PWS":
                    rate = 0.6
                case "SLO":
                    rate = 0.1
        
        # Jam yang tidak terdefinisi menggunakan default rate
        
        PASSENGER_RATES[hour][station_name] = rate

# Destination matrix: probability of going from origin to destination
# Passengers can only travel forward (to next stations) since this is one-way KRL
DESTINATION_PROBS = {}
for i, (origin, _) in enumerate(STATIONS):
    DESTINATION_PROBS[origin] = {}

    # probabilitas perjalanan ke stasiun tujuan - disesuaikan dengan data real kereta 08:49
    if origin == "YK":
        # Dari Yogya naik 656, di Purwosari turun 320, sisanya ke Solo = 296
        # Distribusi turun: LPN(0), MGW(0), KT(40), PWS(320), SLO(296)
        DESTINATION_PROBS[origin] = {
            "LPN": 0.01,  # 1% turun di lempuyangan (sangat sedikit berdasarkan data)
            "MGW": 0.02,  # 2% turun di maguwo (sangat sedikit berdasarkan data)
            "KT": 0.1,   # 6% turun di klaten (40/656 = 6%)
            "PWS": 0.45,  # 42% turun di purwosari (320/656 = 49%, tapi disesuaikan dengan penumpang dari stasiun lain)
            "SLO": 0.42   # 45% sampai solo balapan (296/656 = 45%)
        }
    elif origin == "LPN":
        # Dari pattern yang sama, kebanyakan ke PWS dan SLO
        DESTINATION_PROBS[origin] = {
            "MGW": 0.01,  # 2% turun di maguwo
            "KT": 0.1,   # 8% turun di klaten 
            "PWS": 0.47,  # 50% turun di purwosari (mengikuti pola mayoritas)
            "SLO": 0.42   # 40% sampai solo
        }
    elif origin == "MGW":
        DESTINATION_PROBS[origin] = {
            "KT": 0.10,   # 10% turun di klaten
            "PWS": 0.55,  # 55% turun di purwosari (mengikuti pola mayoritas)
            "SLO": 0.35   # 35% sampai solo
        }
    elif origin == "KT":
        # Dari Klaten naik 208, sebagian ke PWS, sebagian ke SLO
        DESTINATION_PROBS[origin] = {
            "PWS": 0.60,  # 60% turun di purwosari 
            "SLO": 0.40   # 40% lanjut ke solo
        }
    elif origin == "PWS":
        # Dari PWS hanya sedikit naik (12), semuanya ke Solo
        DESTINATION_PROBS[origin] = {
            "SLO": 1.0  # 100% ke Solo karena stasiun terakhir
        }
    elif origin == "SLO":
        DESTINATION_PROBS[origin] = {}  # Tidak ada tujuan dari Solo (stasiun terakhir)

# Parameters
TRAIN_CAPACITY = 1600       # Total train capacity (8 gerbong x 175 penumpang)
SEATED_CAPACITY = 512      # seat tiap gerbong 64 (64 x 8)
TRAIN_SPEED = 1.33         # 80 km/h = 1.33 km/min
BOARDING_TIME = 4         # Minutes for boarding at each station
DWELL_TIME = 2            # Additional time spent at each station

class Passenger:
    def __init__(self, id, origin, destination, arrival_time):
        self.id = id
        self.origin = origin
        self.destination = destination
        self.arrival_time = arrival_time  # Time arrived at the origin station
        self.boarding_time = None         # Time boarded the train
        self.seated = False               # Whether passenger got a seat
        self.completed = False            # Whether journey is completed
        self.train_id = None              # Train the passenger boarded
        self.waiting_at_station = True    # Flag to track if still waiting at station
    
    def __repr__(self):
        return f"Passenger {self.id}: {self.origin} -> {self.destination}"

class Train:
    def __init__(self, id, departure_time, capacity, seated_capacity, simulation):
        self.id = id
        self.departure_time = departure_time
        self.capacity = capacity  # Total capacity (seated + standing)
        self.seated_capacity = seated_capacity  # Number of seats available
        self.passengers = []  # All passengers
        self.seated_passengers = []  # Only seated passengers
        self.standing_passengers = []  # Only standing passengers
        self.current_station_idx = 0
        self.next_station_time = departure_time # Initially, this is the departure time from origin
        self.completed = False
        self.simulation = simulation  # Reference to simulation object for statistics
    
    def _alight_passengers_at_current_station(self):
        current_station_name = self.get_current_station()
        if not current_station_name:
            return

        passengers_to_remove = []
        for passenger in self.passengers:
            if passenger.destination == current_station_name:
                passengers_to_remove.append(passenger)
                passenger.completed = True
                # Note: Overall passenger completion stats are updated in Simulation.update

        for passenger in passengers_to_remove:
            self.passengers.remove(passenger)
            if passenger.seated:
                if passenger in self.seated_passengers: # Check to prevent errors if list modified elsewhere
                    self.seated_passengers.remove(passenger)
                    # When a seat becomes available, give it to the longest-waiting standing passenger
                    if self.standing_passengers:
                        self.standing_passengers.sort(key=lambda p: p.boarding_time)
                        next_to_seat = self.standing_passengers.pop(0)
                        next_to_seat.seated = True
                        self.seated_passengers.append(next_to_seat)
            else:
                if passenger in self.standing_passengers: # Check to prevent errors
                    self.standing_passengers.remove(passenger)

    def board_passengers(self, station_passengers, current_time):
        station = self.get_current_station()
        if not station:
            return station_passengers # Return original list if no current station
        
        boarded_passengers_this_train = []
        passengers_remaining_at_station = list(station_passengers) # Create a copy to modify

        # Get waiting passengers for this station
        # Sort by arrival_time to ensure FIFO for boarding attempts
        waiting_at_this_station = sorted(
            [p for p in passengers_remaining_at_station if p.origin == station and p.waiting_at_station and not p.completed],
            key=lambda p: p.arrival_time
        )
        
        for passenger in waiting_at_this_station:
            if len(self.passengers) >= self.capacity:
                # Train is full, this passenger and subsequent ones cannot board this train
                # They remain in passengers_remaining_at_station with waiting_at_station = True
                # unless they give up later in the main simulation loop.
                break # Stop trying to board passengers on this train
            
            # Board the passenger
            passenger.boarding_time = current_time
            passenger.train_id = self.id
            passenger.waiting_at_station = False  # No longer waiting for THIS train
            self.passengers.append(passenger)
            
            if passenger in passengers_remaining_at_station: # Should always be true here
                 passengers_remaining_at_station.remove(passenger) # Remove from station's general waiting pool

            if len(self.seated_passengers) < self.seated_capacity:
                passenger.seated = True
                self.seated_passengers.append(passenger)
            else:
                passenger.seated = False
                self.standing_passengers.append(passenger)
            
            boarded_passengers_this_train.append(passenger) # Keep track for stats if needed
            
            waiting_time = current_time - passenger.arrival_time
            self.simulation.stats["waiting_times"][self.id].append(waiting_time)
        
        return passengers_remaining_at_station # Return the updated list of passengers waiting at ALL stations

    def _prepare_for_travel_to_next_station(self, arrival_time_at_serviced_station):
        if self.completed:
            return

        # Departure time from the station just serviced (current_station_idx before increment)
        departure_time = arrival_time_at_serviced_station + BOARDING_TIME + DWELL_TIME
        
        self.current_station_idx += 1
        
        if self.current_station_idx >= len(STATIONS):
            self.completed = True
            return
        
        # Calculate travel time to the new STATIONS[self.current_station_idx]
        departing_station_dist = STATIONS[self.current_station_idx - 1][1]
        destination_station_dist = STATIONS[self.current_station_idx][1]
        
        travel_distance = destination_station_dist - departing_station_dist
        # Basic check for valid distance, though STATIONS should be ordered
        travel_distance = max(0, travel_distance) 
            
        travel_time = travel_distance / TRAIN_SPEED
        
        # Set the arrival time for the next station
        self.next_station_time = departure_time + travel_time
    
    def get_current_station(self):
        if self.current_station_idx < len(STATIONS):
            return STATIONS[self.current_station_idx][0]
        return None
    
    def get_next_station(self):
        if self.current_station_idx + 1 < len(STATIONS):
            return STATIONS[self.current_station_idx + 1][0]
        return None
    
    def __repr__(self):
        return f"Train {self.id}: {len(self.passengers)}/{self.capacity} total, {len(self.seated_passengers)}/{self.seated_capacity} seated, {len(self.standing_passengers)} standing"


class Simulation:
    def __init__(self):
        self.current_time = TRAIN_SCHEDULE[0][0] - 60  # Start 1 hour before first train
        self.end_time = MAX_SIMULATION_TIME
        self.clock_speed = SIMULATION_SPEED
        self.passengers = []
        self.passenger_id_counter = 0
        self.trains = []
        self.station_passengers = []
        self.initialize_trains()
        self.stats = {
            "passengers_generated": 0,
            "passengers_completed": 0,
            "passengers_seated": 0,
            "passengers_standing": 0,
            "train_occupancy": defaultdict(list),
            "seated_percentage": defaultdict(list),
            "waiting_times": defaultdict(list),
        }
    
    def initialize_trains(self):
        for i, (departure_time, capacity) in enumerate(TRAIN_SCHEDULE):
            train = Train(i, departure_time, capacity, SEATED_CAPACITY, self)
            self.trains.append(train)
    
    def generate_passenger(self, station, current_hour):
        origin = station
        
        # Only generate if there are valid destinations
        if origin not in DESTINATION_PROBS or not DESTINATION_PROBS[origin]:
            return None
            
        # Determine destination based on origin probabilities
        destinations = list(DESTINATION_PROBS[origin].keys())
        if not destinations:
            return None
            
        probabilities = list(DESTINATION_PROBS[origin].values())
        destination = random.choices(destinations, probabilities)[0]
        
        # Create passenger
        passenger = Passenger(
            self.passenger_id_counter,
            origin,
            destination,
            self.current_time
        )
        self.passenger_id_counter += 1
        self.passengers.append(passenger)
        self.station_passengers.append(passenger)
        self.stats["passengers_generated"] += 1
        
        return passenger
    
    def update(self):
        # Generate passengers at stations based on time of day
        current_hour = (self.current_time // 60) % 24
        
        # Generate passengers only if there are upcoming trains within reasonable time (2 hours)
        for station_name, _ in STATIONS:
            station_idx = next(i for i, (name, _) in enumerate(STATIONS) if name == station_name)
            
            # Check if there are any upcoming trains for this station within 2 hours
            upcoming_trains = [train for train in self.trains 
                             if not train.completed and train.current_station_idx <= station_idx
                             and (train.departure_time + (station_idx * 10)) - self.current_time <= UPCOMING_TRAIN_CHECK_WINDOW]  # Within 2 hours
            
            # Also check if we're still in service hours (before last train + buffer)
            last_train_time = TRAIN_SCHEDULE[-1][0] + LAST_TRAIN_BUFFER
            in_service_hours = self.current_time <= last_train_time
            
            if upcoming_trains and in_service_hours:
                # Get hourly rate for this station
                if current_hour in PASSENGER_RATES and station_name in PASSENGER_RATES[current_hour]:
                    rate = PASSENGER_RATES[current_hour][station_name]
                    
                    # Use Poisson distribution to determine number of new passengers
                    num_new_passengers = np.random.poisson(rate)
                    
                    for _ in range(num_new_passengers):
                        self.generate_passenger(station_name, current_hour)
        
        # Update trains
        for train in self.trains:
            if train.completed:
                continue

            # Handle initial departure from origin
            if self.current_time < train.departure_time and train.current_station_idx == 0 : # Check if it's still before initial departure
                continue
            
            # train.next_station_time is arrival at STATIONS[train.current_station_idx]
            # or initial departure_time if at origin
            if self.current_time >= train.next_station_time:
                
                arrival_time_at_this_station = train.next_station_time 

                # 1. Passengers alight
                # No alighting at the origin station on initial departure
                if not (train.current_station_idx == 0 and arrival_time_at_this_station == train.departure_time):
                    train._alight_passengers_at_current_station()
                
                # 2. Passengers board
                # Boarding uses self.current_time for passenger.boarding_time
                self.station_passengers = train.board_passengers(
                    self.station_passengers, 
                    self.current_time
                )
                
                # 3. Train prepares for travel to the next station
                train._prepare_for_travel_to_next_station(arrival_time_at_this_station)
        
        # Clean up passengers who are no longer waiting and haven't boarded any train
        # Also remove passengers who have been waiting more than PASSENGER_GIVE_UP_WAIT_TIME
        current_waiting_overall = []
        for p in self.station_passengers:
            if p.waiting_at_station and not p.completed and p.boarding_time is None:
                waiting_duration = self.current_time - p.arrival_time
                if waiting_duration <= PASSENGER_GIVE_UP_WAIT_TIME:
                    current_waiting_overall.append(p)
                else:
                    p.waiting_at_station = False # Passenger gives up
        
        self.station_passengers = current_waiting_overall
        
        # Update statistics
        current_active_seated = 0
        current_active_standing = 0
        current_completed_total = 0

        for passenger in self.passengers:
            if passenger.completed:
                current_completed_total += 1
            elif passenger.boarding_time is not None:
                if passenger.seated:
                    current_active_seated += 1
                else:
                    current_active_standing += 1

        self.stats["passengers_completed"] = current_completed_total
        self.stats["passengers_seated"] = current_active_seated
        self.stats["passengers_standing"] = current_active_standing
        
        # Record train occupancy data
        for train in self.trains:
            if not train.completed and self.current_time >= train.departure_time:
                train_id = train.id
                total_passengers = len(train.passengers)
                seated_passengers = len(train.seated_passengers)
                standing_passengers = len(train.standing_passengers)
                
                occupancy_percentage = (total_passengers / train.capacity) * 100
                seated_percentage = (seated_passengers / train.seated_capacity) * 100 if seated_passengers > 0 else 0
                
                self.stats["train_occupancy"][train_id].append((self.current_time, occupancy_percentage))
                self.stats["seated_percentage"][train_id].append((self.current_time, seated_percentage))
        
        # Advance time
        self.current_time += 1
        
        # Check if simulation is complete
        return self.current_time >= self.end_time or all(train.completed for train in self.trains)
    
    def get_results(self):
        return {
            "passengers_generated": self.stats["passengers_generated"],
            "passengers_completed": self.stats["passengers_completed"],
            "passengers_seated": self.stats["passengers_seated"],
            "passengers_standing": self.stats["passengers_standing"],
            "avg_waiting_times": self.calculate_avg_waiting_times(),
            "seat_probability": self.calculate_seat_probability(),
            "seat_probability_yogya": self.calculate_seat_probability_by_origin("YK"),
            "occupancy_data": self.stats["train_occupancy"],
            "seated_percentage": self.stats["seated_percentage"]
        }
    
    def calculate_avg_waiting_times(self):
        waiting_times = {}
        for train_id, times in self.stats["waiting_times"].items():
            if times:
                waiting_times[train_id] = sum(times) / len(times)
        return waiting_times
    
    def calculate_seat_probability(self):
        probabilities = {}
        for train in self.trains:
            completed_passengers = [p for p in self.passengers if p.train_id == train.id and p.completed]
            if completed_passengers:
                seated = len([p for p in completed_passengers if p.seated])
                probabilities[train.id] = seated / len(completed_passengers)
        return probabilities
    
    def calculate_seat_probability_by_origin(self, origin_station="YK"):
        """Calculate seat probability for passengers from specific origin station"""
        probabilities = {}
        for train in self.trains:
            # Filter passengers who boarded from specific origin station
            origin_passengers = [p for p in self.passengers 
                               if p.train_id == train.id and p.completed and p.origin == origin_station]
            if origin_passengers:
                seated = len([p for p in origin_passengers if p.seated])
                probabilities[train.id] = seated / len(origin_passengers)
        return probabilities


class SimulationApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("KRL Jogja-Solo Passenger Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.large_font = pygame.font.SysFont(None, 36)
        self.simulation = Simulation()
        self.running = True
        self.paused = False
        self.fast_forward = False
        self.frame_count = 0
        self.result_graphs = None
        self.recommendations = []
    
    def minutes_to_time_str(self, minutes):
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def station_position(self, station_idx):
        # Calculate station position based on index
        x = 150
        y = 150 + station_idx * 100
        return (x, y)
    
    def render_train(self, train, y_offset=0):
        if train.completed or self.simulation.current_time < train.departure_time:
            return
        
        # Calculate train position
        current_station = train.current_station_idx
        next_station = train.current_station_idx + 1
        
        if next_station >= len(STATIONS):
            # Train at final station
            x, y = self.station_position(current_station)
            x += 100  # Offset from station
        else:
            # Train between stations
            start_x, start_y = self.station_position(current_station)
            end_x, end_y = self.station_position(next_station)
            
            # Calculate progress between stations
            if train.next_station_time > self.simulation.current_time:
                progress = 1 - (train.next_station_time - self.simulation.current_time) / \
                          ((STATIONS[next_station][1] - STATIONS[current_station][1]) / TRAIN_SPEED + DWELL_TIME + BOARDING_TIME)
                progress = max(0, min(1, progress))
            else:
                progress = 1
            
            x = start_x + progress * (end_x - start_x)
            y = start_y + progress * (end_y - start_y)
        
        # Ubah posisi kereta ke kiri
        x = x - 120  # Geser 120 pixel ke kiri dari posisi sekarang
        
        # Draw train
        train_width = 90
        train_height = 40
        train_rect = pygame.Rect(x, y + y_offset - train_height/2, train_width, train_height)
        pygame.draw.rect(self.screen, BLUE, train_rect)
        
        # Add text for train ID and occupancy (sesuaikan posisi text)
        train_id_text = self.font.render(f"KRL{train.id+1}", True, WHITE)
        self.screen.blit(train_id_text, (x + 5, y + y_offset - 10))
        
        # Display occupancy with color based on level
        total_occupancy = len(train.passengers)
        occupancy_pct = total_occupancy / train.capacity * 100
        
        if occupancy_pct < 50:
            color = GREEN
        elif occupancy_pct < 80:
            color = YELLOW
        else:
            color = RED
        
        # Show both total occupancy and breakdown of seated/standing
        occupancy_text = self.font.render(f"{total_occupancy}/{train.capacity}", True, color)
        self.screen.blit(occupancy_text, (x + 5, y + y_offset + 5))
        
        # Show seated/standing counts
        seated = len(train.seated_passengers)
        standing = len(train.standing_passengers)
        seated_text = self.font.render(f"Duduk:{seated}", True, BLACK)
        standing_text = self.font.render(f"Berdiri:{standing}", True, BLACK)
        self.screen.blit(seated_text, (x + 5, y + y_offset + 20))
        self.screen.blit(standing_text, (x + 5, y + y_offset + 35))
    
    def run(self):
        simulation_complete = False
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_f:
                        self.fast_forward = not self.fast_forward
                    elif event.key == pygame.K_r:
                        # Restart simulation
                        self.simulation = Simulation()
                        simulation_complete = False
                        self.result_graphs = None
            
            # Update simulation
            if not self.paused and not simulation_complete:
                # Update multiple times per frame if fast forwarding
                update_count = 10 if self.fast_forward else 1
                
                for _ in range(update_count):
                    simulation_complete = self.simulation.update()
                    
                    if simulation_complete:
                        # Generate results
                        results = self.simulation.get_results()
                        self.result_graphs = self.create_result_graphs(results)
                        self.recommendations = self.generate_recommendations(results)
                        break
            
            # Draw everything
            self.screen.fill(WHITE)
            
            if simulation_complete and self.result_graphs:
                # Display results
                self.screen.blit(self.result_graphs, (0, 0))
                
                # Display recommendations
                y_pos = self.result_graphs.get_height() + 10
                title_text = self.large_font.render("Rekomendasi KRL Jogja-Solo:", True, BLUE)
                self.screen.blit(title_text, (10, y_pos))
                
                for i, recommendation in enumerate(self.recommendations):
                    y_offset = y_pos + 40 + i * 25
                    text = self.font.render(recommendation, True, BLACK)
                    self.screen.blit(text, (20, y_offset))
            else:
                # Draw simulation view
                # Draw stations and tracks
                for i, (station_name, distance) in enumerate(STATIONS):
                    self.render_station(i, station_name, distance)
                    
                    if i < len(STATIONS) - 1:
                        start_x, start_y = self.station_position(i)
                        end_x, end_y = self.station_position(i + 1)
                        pygame.draw.line(self.screen, BLACK, (start_x, start_y), (end_x, end_y), 2)
                
                # Draw trains
                for train in self.simulation.trains:
                    self.render_train(train)
                
                # Display simulation time
                time_str = self.minutes_to_time_str(self.simulation.current_time)
                time_text = self.large_font.render(f"Waktu: {time_str}", True, BLACK)
                self.screen.blit(time_text, (WIDTH - 200, 20))
                
                # Display simulation speed
                speed_multiplier = self.simulation.clock_speed * (10 if self.fast_forward else 1)
                speed_text = self.font.render(f"Kecepatan: {speed_multiplier}x", True, BLACK)
                self.screen.blit(speed_text, (WIDTH - 200, 60))
                
                # Display stats
                stats_text = [
                    f"Total Penumpang: {self.simulation.stats['passengers_generated']}",
                    f"Penumpang Selesai: {self.simulation.stats['passengers_completed']}",
                    f"Penumpang Duduk: {self.simulation.stats['passengers_seated']}",
                    f"Penumpang Berdiri: {self.simulation.stats['passengers_standing']}"
                ]
                
                for i, text in enumerate(stats_text):
                    rendered_text = self.font.render(text, True, BLACK)
                    self.screen.blit(rendered_text, (WIDTH - 300, 100 + i * 30))
                    
                # Tambahkan status layanan
                last_train_time = TRAIN_SCHEDULE[-1][0]
                if self.simulation.current_time > (last_train_time + LAST_TRAIN_BUFFER):
                    service_status = "Layanan KRL Hari Ini Telah Berakhir"
                    status_text = self.font.render(service_status, True, RED)
                    self.screen.blit(status_text, (WIDTH - 300, 220))  # Posisi di bawah stats lainnya
                    
                # Display controls
                controls = [
                    "Kontrol:",
                    "Space - Jeda/Lanjut",
                    "F - Percepat Simulasi",
                    "R - Restart Simulasi"
                ]
                
                for i, text in enumerate(controls):
                    rendered_text = self.font.render(text, True, BLUE)
                    self.screen.blit(rendered_text, (WIDTH - 200, HEIGHT - 120 + i * 25))
            
            pygame.display.flip()
            self.clock.tick(60)
        
        # Show individual station analysis windows after main simulation
        if simulation_complete:
            print("Simulasi selesai! Menampilkan analisis per stasiun...")
            print("Gunakan ESC atau SPACE untuk lanjut ke stasiun berikutnya...")
            
            for station_code, _ in STATIONS:
                station_name = {
                    "YK": "Yogyakarta",
                    "LPN": "Lempuyangan", 
                    "MGW": "Maguwo",
                    "KT": "Klaten",
                    "PWS": "Purwosari",
                    "SLO": "Solo Balapan"
                }.get(station_code, station_code)
                
                print(f"Menampilkan analisis untuk stasiun {station_name}...")
                self.create_station_analysis_window(station_code, station_name)
            
        pygame.quit()
    
    def create_result_graphs(self, results):
        """Create graphs for simulation results"""
        try:
            # Create a set of matplotlib graphs for simulation results
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            
            # Plot 1: Train occupancy over time
            ax1 = axs[0, 0]
            for train_id, data in results["occupancy_data"].items():
                if data and len(data) > 1:  # Ensure we have enough data points
                    times, occupancies = zip(*data)
                    times_adjusted = [(t - self.simulation.trains[train_id].departure_time) / 60 for t in times]  # Convert to hours since departure
                    ax1.plot(times_adjusted, occupancies, label=f"KRL {train_id} ({self.minutes_to_time_str(self.simulation.trains[train_id].departure_time)})")
            
            ax1.set_title("Okupansi Kereta Berdasarkan Waktu")
            ax1.set_xlabel("Jam Sejak Keberangkatan")
            ax1.set_ylabel("Okupansi (%)")
            ax1.grid(True)
            if len(results["occupancy_data"]) > 6:
                # If we have too many trains, make a more compact legend
                ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize='small')
            else:
                ax1.legend()
            
            # Plot 2: Seated percentage over time
            ax2 = axs[0, 1]
            for train_id, data in results["seated_percentage"].items():
                if data and len(data) > 1:  # Ensure we have enough data points
                    times, seated_pcts = zip(*data)
                    times_adjusted = [(t - self.simulation.trains[train_id].departure_time) / 60 for t in times]  # Convert to hours since departure
                    ax2.plot(times_adjusted, seated_pcts, label=f"KRL {train_id} ({self.minutes_to_time_str(self.simulation.trains[train_id].departure_time)})")
            
            ax2.set_title("Persentase Penumpang Duduk")
            ax2.set_xlabel("Jam Sejak Keberangkatan")
            ax2.set_ylabel("Kursi Terisi (%)")
            ax2.grid(True)
            if len(results["seated_percentage"]) > 6:
                # If we have too many trains, make a more compact legend
                ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize='small')
            else:
                ax2.legend()
            
            # Plot 3: Average waiting times by train departure time
            ax3 = axs[1, 0]
            train_ids = []
            wait_times = []
            departure_times = []
            
            for train_id, avg_wait in results["avg_waiting_times"].items():
                train_ids.append(train_id)
                wait_times.append(avg_wait)
                departure_times.append(self.simulation.trains[train_id].departure_time)
            
            # Sort by departure time
            if departure_times:
                sorted_indices = sorted(range(len(departure_times)), key=lambda i: departure_times[i])
                sorted_times = [self.minutes_to_time_str(departure_times[i]) for i in sorted_indices]
                sorted_wait_times = [wait_times[i] for i in sorted_indices]
                
                x = range(len(sorted_times))
                ax3.bar(x, sorted_wait_times, color='orange')
                ax3.set_title("Rata-rata Waktu Tunggu per Keberangkatan")
                ax3.set_xlabel("Waktu Keberangkatan")
                ax3.set_ylabel("Rata-rata Waktu Tunggu (menit)")
                ax3.set_xticks(x)
                ax3.set_xticklabels(sorted_times, rotation=45)
                ax3.grid(True, axis='y')
            
            # Plot 4: Seat probability by train departure time (ALL STATIONS)
            ax4 = axs[1, 1]
            train_ids = []
            seat_probs = []
            departure_times = []
            
            for train_id, seat_prob in results["seat_probability"].items():
                train_ids.append(train_id)
                seat_probs.append(seat_prob * 100)  # Convert to percentage
                departure_times.append(self.simulation.trains[train_id].departure_time)
            
            # Sort by departure time
            if departure_times:
                sorted_indices = sorted(range(len(departure_times)), key=lambda i: departure_times[i])
                sorted_times = [self.minutes_to_time_str(departure_times[i]) for i in sorted_indices]
                sorted_seat_probs = [seat_probs[i] for i in sorted_indices]
                
                x = range(len(sorted_times))
                ax4.bar(x, sorted_seat_probs, color='green')
                ax4.set_title("Probabilitas Mendapatkan Tempat Duduk (Semua Stasiun)")
                ax4.set_xlabel("Waktu Keberangkatan")
                ax4.set_ylabel("Probabilitas Kursi (%)")
                ax4.set_xticks(x)
                ax4.set_xticklabels(sorted_times, rotation=45)
                ax4.grid(True, axis='y')
            
            plt.tight_layout()
            
            # Convert to pygame surface
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.buffer_rgba()
            plt.close(fig)
            
            size = canvas.get_width_height()
            raw_string = raw_data.tobytes()
            
            surf = pygame.image.fromstring(raw_string, size, "RGBA")
            return surf
        except Exception as e:
            print(f"Error creating graphs: {e}")
            return None
    
    def generate_recommendations(self, results):
        # Generate insights and recommendations based on simulation results
        recommendations = []
        
        # Find the best train departures for getting a seat (ALL STATIONS)
        seat_probs = {}
        for train_id, prob in results["seat_probability"].items():
            departure_time = self.simulation.trains[train_id].departure_time
            seat_probs[departure_time] = (train_id, prob)
        
        # Sort by probability (descending)
        sorted_probs = sorted(seat_probs.items(), key=lambda x: x[1][1], reverse=True)
        
        # Best times to get a seat
        best_times = []
        for departure_time, (train_id, prob) in sorted_probs[:3]:
            best_times.append((self.minutes_to_time_str(departure_time), prob * 100))
        
        recommendations.append(f"Waktu keberangkatan terbaik untuk mendapatkan tempat duduk (semua stasiun):")
        for time_str, prob in best_times:
            recommendations.append(f"- {time_str} (probabilitas: {prob:.1f}%)")
        
        # Find the least crowded trains
        train_crowds = {}
        for train_id, data in results["occupancy_data"].items():
            if data:
                # Get average occupancy
                avg_occupancy = sum(occ for _, occ in data) / len(data)
                train_crowds[train_id] = avg_occupancy
        
        # Sort by occupancy (ascending)
        sorted_crowds = sorted(train_crowds.items(), key=lambda x: x[1])
        
        # Least crowded trains
        recommendations.append("\nKereta paling tidak padat:")
        for train_id, avg_occ in sorted_crowds[:3]:
            departure_time = self.simulation.trains[train_id].departure_time
            recommendations.append(f"- {self.minutes_to_time_str(departure_time)} (okupansi rata-rata: {avg_occ:.1f}%)")
        
        # Analyze waiting times
        recommendations.append("\nAnalisis waktu tunggu:")
        
        # Find average waiting time by hour
        waiting_by_hour = defaultdict(list)
        for train_id, waiting_times in self.simulation.stats["waiting_times"].items():
            hour = (self.simulation.trains[train_id].departure_time // 60) % 24
            waiting_by_hour[hour].extend(waiting_times)
        
        # Calculate average waiting time by hour
        avg_waiting_by_hour = {}
        for hour, times in waiting_by_hour.items():
            if times:
                avg_waiting_by_hour[hour] = sum(times) / len(times)
        
        # Find best hours with lowest waiting times
        sorted_waiting = sorted(avg_waiting_by_hour.items(), key=lambda x: x[1])
        
        for hour, avg_wait in sorted_waiting[:3]:
            recommendations.append(f"- Kereta berangkat sekitar jam {hour:02d}:00 memiliki waktu tunggu rata-rata terendah: {avg_wait:.1f} menit")
        
        # Peak hour insights
        recommendations.append("\nInsight jam sibuk:")
        peak_hours = [6, 7, 8, 17, 18]  # Typical peak hours (morning and evening)
        
        for hour in peak_hours:
            if hour in avg_waiting_by_hour:
                recommendations.append(f"- Pada jam {hour:02d}:00 (jam sibuk), waktu tunggu rata-rata adalah {avg_waiting_by_hour[hour]:.1f} menit")
                
                # Get seat probability for trains departing in this hour
                hour_trains = [train_id for train_id, train in enumerate(self.simulation.trains) 
                              if (train.departure_time // 60) % 24 == hour]
                
                hour_seat_probs = []
                for train_id in hour_trains:
                    if train_id in results["seat_probability"]:
                        hour_seat_probs.append(results["seat_probability"][train_id] * 100)
                
                if hour_seat_probs:
                    avg_seat_prob = sum(hour_seat_probs) / len(hour_seat_probs)
                    recommendations.append(f"  Probabilitas mendapat tempat duduk: {avg_seat_prob:.1f}%")
        
        # Add overall recommendations
        recommendations.append("\nKesimpulan dan rekomendasi:")
        recommendations.append(f"- Datang ke stasiun minimal 15-20 menit sebelum keberangkatan")
        recommendations.append(f"- Strategi antrian yang efektif: tunggu di posisi pintu gerbong tengah")
        
        return recommendations
    
    def render_station(self, idx, station_name, distance):
        x, y = self.station_position(idx)
            
        # Draw station circle
        pygame.draw.circle(self.screen, RED, (x, y), 10)
            
        # Draw station name and distance
        station_text = self.font.render(f"{station_name} ({distance}km)", True, BLACK)
        self.screen.blit(station_text, (x + 20, y - 10))
            
        # Count only passengers who are actively waiting at this station
        waiting = sum(1 for p in self.simulation.station_passengers 
                     if p.origin == station_name and p.waiting_at_station)
            
        # Draw waiting passengers indicator
        waiting_text = self.font.render(f"Tunggu: {waiting}", True, RED if waiting > 50 else BLACK)
        self.screen.blit(waiting_text, (x + 20, y + 10))

    def create_station_analysis_window(self, station_code, station_name):
        """Create a separate window for individual station analysis"""
        pygame.display.set_caption(f"Analisis Stasiun {station_name} ({station_code})")
        
        # Get results for this specific station
        results = self.simulation.get_results()
        station_graphs = self.create_station_specific_graphs(station_code, station_name, results)
        
        if not station_graphs:
            print(f"Error: Tidak bisa membuat grafik untuk stasiun {station_name}")
            return
        
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return  # Exit this station analysis
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE:
                        running = False  # Move to next station
            
            self.screen.fill(WHITE)
            
            # Display the station-specific graphs
            self.screen.blit(station_graphs, (0, 0))
            
            # Add instructions
            instruction_text = self.font.render("Press ESC atau SPACE untuk lanjut ke stasiun berikutnya", True, BLUE)
            self.screen.blit(instruction_text, (10, HEIGHT - 30))
            
            pygame.display.flip()
            self.clock.tick(60)
    
    def create_station_specific_graphs(self, station_code, station_name, results):
        """Create graphs specific to one station"""
        try:
            print(f"Creating graphs for {station_name}...")
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(f"Analisis Detil Stasiun {station_name} ({station_code})", fontsize=16)
            
            # Plot 1: Seat probability for this station
            ax1 = axs[0, 0]
            station_seat_probs = self.simulation.calculate_seat_probability_by_origin(station_code)
            
            if station_seat_probs:
                train_ids = []
                seat_probs = []
                departure_times = []
                
                for train_id, seat_prob in station_seat_probs.items():
                    train_ids.append(train_id)
                    seat_probs.append(seat_prob * 100)
                    departure_times.append(self.simulation.trains[train_id].departure_time)
                
                if departure_times:
                    sorted_indices = sorted(range(len(departure_times)), key=lambda i: departure_times[i])
                    sorted_times = [self.minutes_to_time_str(departure_times[i]) for i in sorted_indices]
                    sorted_seat_probs = [seat_probs[i] for i in sorted_indices]
                    
                    x = range(len(sorted_times))
                    ax1.bar(x, sorted_seat_probs, color='green')
                    ax1.set_title(f"Probabilitas Tempat Duduk dari {station_name}")
                    ax1.set_xlabel("Waktu Keberangkatan")
                    ax1.set_ylabel("Probabilitas Kursi (%)")
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(sorted_times, rotation=45)
                    ax1.grid(True, axis='y')
                else:
                    ax1.text(0.5, 0.5, 'Tidak ada data keberangkatan', ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title(f"Probabilitas Tempat Duduk dari {station_name}")
            else:
                ax1.text(0.5, 0.5, 'Tidak ada data penumpang', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f"Probabilitas Tempat Duduk dari {station_name}")
            
            # Plot 2: Passenger generation rate by hour for this station
            ax2 = axs[0, 1]
            hours = list(range(24))
            rates = [PASSENGER_RATES[hour].get(station_code, 0) for hour in hours]
            
            ax2.plot(hours, rates, marker='o', linewidth=2, markersize=4, color='blue')
            ax2.set_title(f"Rate Penumpang per Jam - {station_name}")
            ax2.set_xlabel("Jam")
            ax2.set_ylabel("Penumpang/Menit")
            ax2.grid(True)
            ax2.set_xticks(range(0, 24, 2))
            
            # Plot 3: Destination distribution from this station
            ax3 = axs[1, 0]
            if station_code in DESTINATION_PROBS and DESTINATION_PROBS[station_code]:
                destinations = list(DESTINATION_PROBS[station_code].keys())
                probabilities = list(DESTINATION_PROBS[station_code].values())
                
                colors = plt.cm.Set3(range(len(destinations)))
                ax3.pie(probabilities, labels=destinations, autopct='%1.1f%%', startangle=90, colors=colors)
                ax3.set_title(f"Distribusi Tujuan dari {station_name}")
            else:
                ax3.text(0.5, 0.5, 'Tidak ada tujuan\n(stasiun terakhir)', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title(f"Distribusi Tujuan dari {station_name}")
            
            # Plot 4: Summary statistics
            ax4 = axs[1, 1]
            ax4.axis('off')
            
            # Calculate some stats for this station
            station_passengers = [p for p in self.simulation.passengers if p.origin == station_code]
            total_generated = len(station_passengers)
            completed = len([p for p in station_passengers if p.completed])
            seated = len([p for p in station_passengers if p.completed and p.seated])
            
            stats_text = [
                f"Total Penumpang: {total_generated}",
                f"Perjalanan Selesai: {completed}",
                f"Mendapat Tempat Duduk: {seated}",
                f"Tingkat Keberhasilan: {(completed/total_generated*100):.1f}%" if total_generated > 0 else "Tingkat Keberhasilan: 0%",
                f"Rate Tempat Duduk: {(seated/completed*100):.1f}%" if completed > 0 else "Rate Tempat Duduk: 0%"
            ]
            
            for i, text in enumerate(stats_text):
                ax4.text(0.1, 0.8 - i*0.15, text, fontsize=12, fontweight='bold', color='darkblue')
            
            ax4.set_title(f"Statistik {station_name}")
            
            plt.tight_layout()
            
            # Convert to pygame surface
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.buffer_rgba()
            plt.close(fig)
            
            size = canvas.get_width_height()
            raw_string = raw_data.tobytes()
            
            surf = pygame.image.fromstring(raw_string, size, "RGBA")
            print(f"Successfully created graphs for {station_name}")
            return surf
            
        except Exception as e:
            print(f"Error creating graphs for {station_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    app = SimulationApp()
    app.run()