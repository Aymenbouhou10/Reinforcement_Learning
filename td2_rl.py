import base
import numpy as np


def greedy(q):
    p = np.zeros(q.shape)
    for s in range(p.shape[0]):
        p[s, np.argmax(q[s])] = 1
    return p

def egreedy(q, epsilon):
    ns = q.shape[0]
    na = q.shape[1]
    p = np.zeros(q.shape) + epsilon / na 
    for s in range(ns):
        p[s, np.argmax(q[s])] += 1 - epsilon
    return p
    

def monte_carlo_control(env, gamma, nb_ep_limit):
    na = env.get_nb_actions()
    q = np.zeros((env.get_nb_states(), na))
    n = np.zeros((env.get_nb_states(), na))
    epsilon = 1
    pi = egreedy(q, epsilon)
    for k in range(1, nb_ep_limit+1):
        # Faire un episode
        s = env.reset()
        episode = []
        while not env.is_final(s):
            a = np.random.choice(na, p=pi[s]) 
            ns, r, _, _ = env.step(a)
            episode.append([s, a, ns, r])
            s = ns
            
        # Mettre à jour Q
        retour = 0
        for s, a, ns, r in reversed(episode):
            retour = r + gamma * retour
            n[s, a] += 1
            q[s, a] += 1/n[s, a]*(retour - q[s, a])
       
        # Mettre à jour epsilon et pi
        epsilon = 1 / k
        pi = egreedy(q, epsilon)
        
    return pi

def egreedy_decision(qvalues, state, epsilon):
    if np.random.rand()<epsilon:
        return np.random.randint(0, qvalues.shape[1])
    else:
        return np.argmax(qvalues[state])

def qlearning(env, gamma, alpha, epsilon, nb_ep_limit):
    na = env.get_nb_actions()
    q = np.zeros((env.get_nb_states(), na))
    for k in range(1, nb_ep_limit+1):      
        s = env.reset()
        while not env.is_final(s):
            a = egreedy_decision(q, s, epsilon)
            ns, r, _, _ = env.step(a)
            q[s, a] += alpha * (r + gamma * np.max(q[ns]) - q[s, a])
            s = ns
    
    return greedy(q)
    
            
        
        
        
env = base.Maze()

pi = monte_carlo_control(env, gamma, epsilon)
#pi = sarsa(env, gamma, alpha, epsilon)
#pi = qlearning(env, gamma, alpha, epsilon)

env.observe_episode(pi, 10)