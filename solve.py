import random
import sys
import matplotlib.pylab as plt
import hardest_game
import numpy as np

preprocess = {}
pps = {}


def play_game_AI(str, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AI').run_AI_moves_graphic(moves=str)
    return game


def simulate(str, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AI').run_AI_moves_no_graphic(moves=str)
    return game


def run_whole_generation(list_of_strs, N, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AIS').run_generation(list_of_moves=list_of_strs, move_len=N)
    return game


def play_human_mode(map_name='map1.txt'):
    hardest_game.Game(map_name=map_name, game_type='player').run_player_mode()


def give_birth_to_random_initial_gens():
    genomes = ['w', 's', 'd', 'a', 'x']
    gens = []
    for i in range(2500):
        initial_gen = ''
        for j in range(20):
            initial_gen += genomes[random.randint(0, 3)]
        gens.append(initial_gen)
    return np.array(gens)


def get_fitness_of_generation(game):
    global preprocess
    is_dead = []
    fitness = np.sum(np.array(game.goal_player).astype('float64'), axis=0) * 700
    for i in range(len(game.players)):
        individual = game.players[i]
        if individual[2]:
            return [-1, i]
        ratio = 1
        is_dead.append(individual[1])
        if individual[1] != -1:
            ratio = 0.8
        goals = game.goals
        min_dis = sys.maxsize
        my_goal = None
        for j in range(len(goals)):
            goal = goals[j]
            if game.goal_player[j][i]:
                fitness[i] += 100
                continue
            distance = np.sqrt((individual[0].x - goal[0].x) ** 2 + (individual[0].y - goal[0].y) ** 2)
            if distance < min_dis:
                my_goal = goal
                preprocess = pps[(goal[0].x, goal[0].y)]
                min_dis = distance
        if my_goal is None:
            my_goal = game.end
            d = get_bfs_depth(individual[0].x, individual[0].y)
            fitness[i] += 700 - np.sqrt(((individual[0].x - (my_goal.x + my_goal.w)) ** 2 + (
                    individual[0].y - (my_goal.y + my_goal.h / 4)) ** 2)) + d
        else:
            fitness[i] += 700 - min_dis
            fitness[i] *= ratio

    return fitness, np.array(is_dead)


def natural_selection(population, fittness, is_dead):
    new_is_dead = []
    max_fitness = np.max(fittness)
    args = fittness.argsort()[len(fittness) - 5:]
    min_fitness_of_top20 = np.min(fittness[args])
    new_population = []
    new_fitness = []
    t20_counter = 0
    for i in range(len(population)):
        if fittness[i] == 0:
            continue
        random_var = random.random()
        var = np.exp(5 * (fittness[i] - max_fitness) / max_fitness)
        if t20_counter < 20:
            if fittness[i] >= min_fitness_of_top20:
                new_population.append(population[i])
                new_fitness.append(fittness[i])
                new_is_dead.append(is_dead[i])
                t20_counter += 1
                continue
        if random_var <= var:
            new_population.append(population[i])
            new_fitness.append(fittness[i])
            new_is_dead.append(is_dead[i])
    return np.array(new_population), np.array(new_fitness), np.array(new_is_dead)


def add_some_genome(population):
    genomes = ['w', 's', 'd', 'a', 'x']
    new_population = []
    for i in range(len(population)):
        new_man = str(population[i])
        for j in range(1):
            new_man += str(genomes[random.randint(0, 4)])
        new_population.append(new_man)
    return np.array(new_population)


def change_some_genome(population):
    new_population = []
    for i in range(len(population)):
        new_man = str(population[i])[:-30]
        new_population.append(new_man)
    return new_population


def fill(population, fitness, is_dead):
    p = len(population)
    arg = np.argpartition(fitness, -5)[-5:]
    top5 = population[arg]
    top5_is_dead = is_dead[arg]
    j = 0
    l = []
    id = []
    for i in range(2500 - p):
        l.append(top5[j])
        id.append(top5_is_dead[j])
        if j == 4:
            j = 0
        else:
            j += 1
    return np.concatenate((population, np.array(l))), np.concatenate((is_dead, np.array(id)))


def get_bfs_depth(x, y):
    try:
        return preprocess[(x, y)]
    except KeyError:
        return 0


def ok(point, direction, game):
    v_lines = game.Vlines
    h_lines = game.Hlines
    if direction == 'vf':
        for h_line in h_lines:
            if point[1] < h_line.y1 < point[1] + 5 and h_line.x1 < point[0] < h_line.x2:
                return False
    elif direction == 'vb':
        for h_line in h_lines:
            if point[1] - 5 < h_line.y1 < point[1] and h_line.x1 < point[0] < h_line.x2:
                return False
    elif direction == 'hf':
        for v_line in v_lines:
            if point[0] < v_line.x1 < point[0] + 5 and v_line.y1 < point[1] < v_line.y2:
                return False
    elif direction == 'hb':
        for v_line in v_lines:
            if point[0] - 5 < v_line.x1 < point[0] and v_line.y1 < point[1] < v_line.y2:
                return False
    return True


def bfs(x, y, game):
    global preprocess
    preprocess = {}
    q = [(x, y)]
    depth = [0]
    end = game.end
    while q:
        current_point = q.pop(0)
        d = depth.pop(0)
        if end.x < current_point[0] < end.x + end.w and end.y < current_point[1] < end.y + end.h:
            return
        preprocess[current_point] = d
        if ok(current_point, 'vf', game) and \
                (current_point[0] + 5, current_point[1]) not in preprocess.keys() and \
                (current_point[0] + 5, current_point[1]) not in q:
            q.append((current_point[0] + 5, current_point[1]))
            depth.append(d + 1)
        if ok(current_point, 'vb', game) and \
                (current_point[0], current_point[1] + 5) not in preprocess.keys() and \
                (current_point[0], current_point[1] + 5) not in q:
            q.append((current_point[0], current_point[1] + 5))
            depth.append(d + 1)
        if ok(current_point, 'hf', game) and \
                (current_point[0] - 5, current_point[1]) not in preprocess.keys() and \
                (current_point[0] - 5, current_point[1]) not in q:
            q.append((current_point[0] - 5, current_point[1]))
            depth.append(d + 1)
        if ok(current_point, 'hb', game) and \
                (current_point[0], current_point[1] - 5) not in preprocess.keys() and \
                (current_point[0], current_point[1] - 5) not in q:
            q.append((current_point[0], current_point[1] - 5))
            depth.append(d + 1)


def make_a_child(father, mother, is_father_dead, is_mother_dead):
    child = ''
    for i in range(len(father)):
        last_40_percent = len(father) // 2
        random_var = random.random()
        if random_var > 0.5:
            if is_mother_dead == -1 or i < last_40_percent:
                child += mother[i]
            else:
                child += father[i]
        else:
            if is_father_dead == -1 or i < last_40_percent:
                child += father[i]
            else:
                child += mother[i]
    return child


def mate(population, is_dead):
    children = []
    for i in range(len(population)):
        if i == 2499:
            child = population[i]
        else:
            parent1 = population[i]
            parent2 = population[i + 1]
            child = make_a_child(parent1, parent2, is_dead[i], is_dead[i + 1])
        children.append(child)
    return children


def mutate(population):
    genomes = ['w', 's', 'd', 'a', 'x']
    new_population = []
    for i in range(len(population)):
        gen = population[i]
        random_list = []
        random_var = random.random()
        if random_var < 0.2:
            random_mutation = ''
            for j in range(len(population[0]) // 5):
                random_mutation += str(genomes[random.randint(0, 4)])
            gen = gen[0:len(population[0]) - len(population[0]) // 5] + random_mutation
        new_population.append(gen)
    return new_population


def genetic_algorithm(map):
    fittest_values = {}
    generation_number = 0
    gens = give_birth_to_random_initial_gens()
    number_of_genomes = 20
    result_game = run_whole_generation(gens, number_of_genomes, map)
    for goal in result_game.goals:
        goal = goal[0]
        bfs(goal.x, goal.y, result_game)
        pps[(goal.x, goal.y)] = preprocess
    while True:
        result_game = run_whole_generation(gens, number_of_genomes, map)
        fitness, is_dead = get_fitness_of_generation(result_game)
        if isinstance(fitness, int):
            print(gens[is_dead])
            play_game_AI(gens[is_dead], map)
            return fittest_values
        elite, fitness, is_dead = natural_selection(gens, fitness, is_dead)
        elite, is_dead = fill(elite, fitness, is_dead)
        arg = np.random.permutation(len(elite))
        elite = elite[arg]
        is_dead = is_dead[arg]
        elite = mate(elite, is_dead)
        if np.sum(is_dead == -1) < 20:
            elite = change_some_genome(elite)
            number_of_genomes = len(elite[0])
        else:
            elite = mutate(elite)
            elite = add_some_genome(elite)
            number_of_genomes += 1
        gens = elite
        generation_number += 1
        fittest_values[generation_number] = max(fitness)


if __name__ == '__main__':
    f = genetic_algorithm('map3.txt')
    myList = f.items()
    myList = sorted(myList)
    x, y = zip(*myList)
    plt.plot(x, y)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Change Over Time (Map3)')
    plt.show()
