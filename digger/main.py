"""
Runs the games, trains the model and saves the weights

within n_battles=7000 the model trains fairly well to simple move-to-target task
using N_HIDDEN_LAYERS = 2, HIDDEN_DIM = 16, BATCH_SIZE = 32, BUFFER_SIZE = 512
"""
import digger.utils as dutils
import digger.plotting as dplotting
import digger.constants as dconst
import digger.structures as dstruct
import digger.fighters as dfighters
import digger.battlefield as dbattlefield
from digger.terrain import terrains
import digger.terrain as dterrain


if __name__ == "__main__":
    # config
    n_battles = 7000
    is_train = True

    # make sure the folders exist
    dutils.maybe_make_dir(dconst.MODELS_FOLDER)

    # define fighters
    fighters = dfighters.Fighters()
    fighters.add_fighter(fighter_class='arkonaut_volley', name='Gorodrin', player_id=0)
    # fighters.add_fighter(fighter_class="mizzenmaster", name="Gordur", player_id=0)

    # create and reset battlefield
    battlefield = dbattlefield.Battlefield(fighters=fighters, players=[0], terrains=terrains)
    battlefield.reset(fighters)

    # hardcode single_fighter for now
    single_fighter = fighters.fighters[0]

    # either load existing model or pre-populate buffer with random moves
    if is_train:
        # collect random moves to fit the scaler and prefill the buffer
        single_fighter.collect_random_moves(battlefield, fighters)
    else:
        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        single_fighter.epsilon = 0.01

        # load agent including trained weights and scaler
        # NOTE JB: scaler is currently part of fighter and is saved as part of agent
        single_fighter.load(f"{dconst.MODELS_FOLDER}/dqn.ckpt")

    # clean the logs as there can be entries from initial random moves
    battlefield.action_log.clean_log()
    battlefield.battle_log.clean_log()

    # main cycle - play the game n_battles times
    for e in range(n_battles):
        battlefield.play_battle(fighters, num_rounds=dconst.ROUNDS_PER_BATTLE, battle_number=e)

    # save the agent
    if is_train:
        # save the DQN
        single_fighter.save(f"{dconst.MODELS_FOLDER}/dqn.ckpt")

    # AFTER RUN CHECKS
    # see battle log
    battle_log = battlefield.battle_log.get_log()

    # plotting
    n_plots = 5
    fld_img = dutils.create_timestamp_folder(dconst.FLD_IMG_RESULTS)

    # plot rewards
    dplotting.plot_rewards(battlefield, save_fld=fld_img)

    # plot several successful paths
    idx_battles_success = battle_log.loc[battle_log["fighter_reward"] > 0, "battle_number"][:n_plots].tolist()
    for i in idx_battles_success:
        dplotting.plot_actions(fighters, battlefield, terrains, buffer=0, battle_number=i, save_fld=fld_img)

    # plot several other paths
    idx_battles_all = list(range(n_battles - n_plots + 1, n_battles + 1))
    for i in idx_battles_all:
        dplotting.plot_actions(fighters, battlefield, terrains, buffer=0, battle_number=i, save_fld=fld_img)
    print("Plotting done")

