"""
Plotting functions for digger package.
"""
from shapely.plotting import plot_polygon, plot_line, plot_points
import matplotlib.pyplot as plt
import matplotlib


def plot_terrains_and_lines(terrains=None, lines=None, point=None):
    """Plots terrains and fighters' move paths. Optionally plots a point. e.g. target."""
    if point is not None:
        plt.scatter(point[0], point[1], marker="x", c="black")

    if terrains is not None:
        for terrain in terrains:
            plot_polygon(terrain)

    if lines is not None:
        for line in lines:
            plot_line(line)
    plt.show()


def plot_actions(fighters, battlefield, terrains, battle_number=None, buffer=0.98):
    """Plot actions of fighters in given battle. Optionally buffer terrains."""
    log = battlefield.action_log.get_log(battle_number)
    battle_log = battlefield.battle_log.get_log(battle_number)
    cmap = matplotlib.colormaps["Spectral"]
    fig, ax = plt.subplots()

    for terrain in terrains:
        plot_polygon(terrain.buffer(buffer, quad_segs=1))

    for f_id in fighters.fighter_ids:
        color_now = cmap(f_id / 7)
        fighter_now = fighters.fighters[f_id]
        # f_id = 0
        positions = (
            log
            .loc[log['fighter_id'] == f_id, :]
            .sort_values(by=['round', 'action_order'])[['position_start', 'position_end']]
            .reset_index(drop=False, names='orig_index')
        )
        if positions.shape[0] == 0:
            break

        points = positions['position_start'].to_list()
        points.append(positions['position_end'].iloc[-1])

        for point in points:
            circle_now = plt.Circle(point, fighter_now.base / 2, color=color_now, fill=False)
            ax.add_patch(circle_now)

        for row in positions.iterrows():
            row = row[1]
            path_now = battlefield.move_paths[row['orig_index']]
            plot_line(path_now, color=color_now)

    # plot target point from battle_log
    if battle_number is not None:
        target = battle_log.loc[battle_log['battle_number'] == battle_number, 'target_position'].iloc[0]
        plt.scatter(target[0], target[1], marker="x", c="black")

    plt.show()


def plot_rewards(battlefield):
    """Plot rewards in each battle."""
    # get data from battle log
    battle_log = battlefield.battle_log.get_log()
    battle_numbers = battle_log['battle_number'].ravel()
    fighter_rewards = battle_log['fighter_reward'].ravel()

    # plot
    plt.scatter(battle_numbers, fighter_rewards, alpha=0.5, linewidths=0)
    plt.xlabel('Battle number')
    plt.ylabel('Fighter reward')
    plt.show()
