import environments
import vmas

def get_environment(name:str, envs:int, agents:int, device:str, grad_enabled:bool, seed:int, is_eval:bool=False)->vmas.simulator.environment.Environment:
    
    match name:
        case "dispersion" :
            return vmas.simulator.environment.Environment(
                environments.scenarios.Dispersion(
                    device = device ,
                    radius = .05    ,
                    agents = agents ,
                ),
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )

        case "transport" :
            return vmas.simulator.environment.Environment(
                environments.scenarios.Transport(),
                package_mass       = 10.0         ,
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )

        case "sampling" :
            return vmas.simulator.environment.Environment(
                environments.scenarios.Sampling() ,
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )

        case "reverse_transport" :
            return vmas.simulator.environment.Environment(
                environments.scenarios.ReverseTransport(),
                package_mass       = 10.0         ,
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )

        case "flocking" :
            return vmas.simulator.environment.Environment(
                environments.scenarios.Flocking(),
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )

        case "discovery" :
            return vmas.simulator.environment.Environment(
                environments.scenarios.Discovery(
                    agents_per_target = min(agents, 2),
                ),
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )
        
        case "pendulum":
            return vmas.make_env(
                scenario=environments.scenarios.Pendulum(),
                num_envs=envs,
                device=device,
                continuous_actions=True,
                seed=seed,
                grad_enabled=grad_enabled,
                render_mode=None,
                n_agents=1,
                action_size=1,  # 1 continuous action: torque
            )
        
        case "breakout":
            return vmas.make_env(
                scenario=environments.scenarios.Breakout(),
                num_envs=envs,
                device=device,
                continuous_actions=True,
                seed=seed,
                grad_enabled=False,
                feature_dim=512,
                gym_env_name="ALE/Breakout-v5",
                render_mode=None
            )
        
        case "breakout_ram":
            return vmas.make_env(
                scenario=environments.scenarios.Breakout_Ram(),
                num_envs=envs,
                device=device,
                continuous_actions=True,
                seed=seed,
                grad_enabled=False,
                render_mode=None,
                n_agents=1,
                action_size=2,  # 2 continuous actions: LEFT, RIGHT
                life_loss_penalty=0,
                tracking_bonus=0.5,
                direction_bonus=0,
            )

        case _:
            raise ValueError(f"Unknown environment {name}.")
