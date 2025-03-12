#!/usr/bin/env python

from project2_base import *
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import random
import cartopy.crs as ccrs
from traffic.core import Flight
from geopy.distance import distance

# Global storage for flight data
ground_truth_flights = None
radar_flights = None

def load_flights():
    """Loads ground truth and radar flight data"""
    global ground_truth_flights, radar_flights
    if ground_truth_flights is None or radar_flights is None:
        ground_truth_flights = get_ground_truth_data()
        radar_flights = get_radar_data(ground_truth_flights)
    print("Flight data loaded.")

def plot_flights():
    """Tasks 2,3: Plot a sample of ground truth and radar flights."""
    flight_ids = random.sample(list(ground_truth_flights.keys()), 6)
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(3, 2, figsize=(12, 18), subplot_kw={'projection': projection})

    for ax, flight_id in zip(axes.flat, flight_ids):
        ax.coastlines()
        ax.set_title(f"Flight {flight_id}")
        ground_truth_flights[flight_id].plot(ax=ax, transform=projection, color='blue', label="Ground Truth")
        radar_flights[flight_id].plot(ax=ax, transform=projection, color='red', linestyle='dashed', label="Radar")
        ax.legend()

    plt.savefig("flight_paths.png", dpi=300, bbox_inches='tight')
    plt.show()

def run_kalman_filter(flight_id):
    """Tasks 1 and 4: Runs a Kalman filter on a selected flight and returns estimated states."""
    my_flight = radar_flights[flight_id]
    radar_measurements = np.array(list(zip(my_flight.data['x'], my_flight.data['y'])))

    d_t = 10  # 10s between radar measurements
    F_k = np.array([[1, 0, d_t, 0 ],
                    [0, 1, 0, d_t],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])    

    Q_k = np.zeros((4, 4))
    sigma_p = 1.5
    var_pos = sigma_p**2 * d_t**4 / 4
    var_vel = sigma_p**2 * d_t**2
    cov_pos_vel = sigma_p**2 * d_t**3 / 2   
    Q_k[0, 0] = var_pos
    Q_k[1, 1] = var_pos
    Q_k[2, 2] = var_vel
    Q_k[3, 3] = var_vel
    Q_k[0, 2] = Q_k[2, 0] = cov_pos_vel
    Q_k[1, 3] = Q_k[3, 1] = cov_pos_vel

    H_k = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])

    R_k = np.zeros((2, 2))
    sigma_o = 100
    R_k[0, 0] = R_k[1, 1] = sigma_o**2  

    x_0 = np.array([radar_measurements[0, 0], radar_measurements[0, 1], 0, 0])
    P_0 = np.array([[5000, 0, 0, 0],
                    [0, 5000, 0, 0],
                    [0, 0, 100, 0],
                    [0, 0, 0, 100]])

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = x_0
    kf.P = P_0
    kf.F = F_k
    kf.H = H_k
    kf.Q = Q_k
    kf.R = R_k

    estimated_states = []
    for z_k in radar_measurements:
        kf.predict()
        kf.update(z_k)
        estimated_states.append((kf.x[0], kf.x[1]))

    estimated_states = np.array(estimated_states)

    filtered_data = my_flight.data.iloc[:len(estimated_states)].copy()
    filtered_data['x'], filtered_data['y'] = zip(*estimated_states)
    filtered_flight = set_lat_lon_from_x_y(Flight(filtered_data))

    return my_flight, filtered_flight, estimated_states, radar_measurements

def plot_kalman_results(flight_id, my_flight, filtered_flight):
    """Task 5: Plots Kalman-filtered results against radar and ground truth."""
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': projection})
    ax.coastlines()

    ground_truth_flights[flight_id].plot(ax=ax, transform=projection, color='blue', label="Ground Truth")
    my_flight.plot(ax=ax, transform=projection, color='red', linestyle='dotted', linewidth=2, label="Radar")
    filtered_flight.plot(ax=ax, transform=projection, color='lime', linestyle='dashed', linewidth=1.5, label="Filtered")

    ax.legend()
    plt.savefig("filtered_paths.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_residuals(estimated_states, radar_measurements):
    """Plots residuals (difference between radar and filtered data). Used as filter check"""
    residuals = radar_measurements - estimated_states
    residuals_lon = residuals[:, 0]
    residuals_lat = residuals[:, 1]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(residuals_lon, label="Longitude Residuals", color="red")
    ax[0].set_ylabel("Longitude Difference (meters)")
    ax[0].legend()
    ax[1].plot(residuals_lat, label="Latitude Residuals", color="blue")
    ax[1].set_ylabel("Latitude Difference (meters)")
    ax[1].set_xlabel("Time Step")
    ax[1].legend()

    plt.suptitle("Residuals: Radar vs. Filtered Flight")
    plt.show()

    

def compute_flight_errors(flight_id, ground_truth_flights, filtered_flight): #filtered_flights):
    """Compute mean and max error for a single flight."""
    gt_data = ground_truth_flights[flight_id].data
    filtered_data = filtered_flight.data #filtered_flights[flight_id].data

    # Ensure both datasets have the same length
    min_length = min(len(gt_data), len(filtered_data))
    gt_data = gt_data.iloc[:min_length]
    filtered_data = filtered_data.iloc[:min_length]

    # Compute errors
    errors = [
        distance((gt_row.latitude, gt_row.longitude), (filt_row.latitude, filt_row.longitude)).meters
        for gt_row, filt_row in zip(gt_data.itertuples(), filtered_data.itertuples())
    ]

    return {"mean_error": sum(errors) / len(errors), "max_error": max(errors)}

if __name__ == "__main__":
    load_flights()
    #print(ground_truth_flights["VHOMS_054"].data.head())  # Before radar_data
    #print(radar_flights["VHOMS_054"].data.head())  # After radar_data

    # Plot ground truth and radar flights:
    #plot_flights()  # View raw data first

    selected_flight_id = random.choice(list(ground_truth_flights.keys()))
    my_flight, filtered_flight, estimated_states, radar_measurements = run_kalman_filter(selected_flight_id)

    plot_kalman_results(selected_flight_id, my_flight, filtered_flight)
    plot_residuals(estimated_states, radar_measurements)
    print(compute_flight_errors(selected_flight_id, ground_truth_flights, filtered_flight))