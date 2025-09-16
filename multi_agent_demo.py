#!/usr/bin/env python3
"""
DiffPhysDrone Multi-Agent Demo Script
=====================================

A simplified demonstration script for running multi-agent drone simulations
with obstacles. This can be run locally or adapted for different environments.

Features:
- Multi-agent swarm simulation
- Dynamic obstacle generation
- Vision-based control
- Real-time visualization

Usage:
    python multi_agent_demo.py [--batch_size 32] [--num_iters 1000] [--single] [--no_gates]
"""

import argparse
import math
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Import modules (ensure they're in path)
try:
    from env_cuda import Env
    from model import Model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you've built the CUDA extensions with: pip install -e src/")
    exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DiffPhysDrone Multi-Agent Demo")
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Number of parallel simulations (default: 32)')
    parser.add_argument('--num_iters', type=int, default=1000,
                       help='Number of training iterations (default: 1000)')
    parser.add_argument('--timesteps', type=int, default=100,
                       help='Simulation timesteps per iteration (default: 100)')
    parser.add_argument('--single', action='store_true',
                       help='Use single agent instead of multi-agent')
    parser.add_argument('--no_gates', action='store_true',
                       help='Disable gate obstacles')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')
    parser.add_argument('--save_model', type=str, default='demo_model.pth',
                       help='Path to save trained model (default: demo_model.pth)')
    return parser.parse_args()


def setup_device(device_arg):
    """Setup computation device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    return device


def create_environment(args, device):
    """Create simulation environment."""
    env = Env(
        batch_size=args.batch_size,
        width=64,
        height=48,
        grad_decay=0.4,
        device=device,
        fov_x_half_tan=0.82,
        single=args.single,
        gate=not args.no_gates,
        ground_voxels=False,
        scaffold=False,
        speed_mtp=1.0,
        random_rotation=False,
        cam_angle=10
    )
    
    print(f"Environment created:")
    print(f"  Multi-agent: {not args.single}")
    print(f"  Gate obstacles: {not args.no_gates}")
    print(f"  Batch size: {args.batch_size}")
    
    return env


def create_model(device):
    """Create and initialize model."""
    model = Model(7+3, 6)  # With odometry
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def visualize_state(env, step=0, save_fig=None):
    """Create visualization of current environment state."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Render depth image
    depth, _ = env.render(1/15)
    depth_vis = depth[0].cpu().numpy()
    depth_vis = np.clip(depth_vis / 10.0, 0, 1)
    
    axes[0, 0].imshow(depth_vis, cmap='viridis')
    axes[0, 0].set_title('Agent Depth View')
    axes[0, 0].axis('off')
    
    # 3D positions
    ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
    
    # Current and target positions
    n_agents = min(env.n_drones_per_group, 8)  # Limit for visualization
    pos = env.p[:n_agents].cpu().numpy()
    target_pos = env.p_target[:n_agents].cpu().numpy()
    
    ax_3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
                  c='blue', s=50, label='Current Position')
    ax_3d.scatter(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 
                  c='red', s=50, marker='x', label='Target Position')
    
    # Show some obstacles
    balls = env.balls[0].cpu().numpy()
    for i, ball in enumerate(balls[:5]):
        if ball[3] > 0:
            ax_3d.scatter(ball[0], ball[1], ball[2], 
                         c='gray', s=ball[3]*200, alpha=0.5)
    
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y') 
    ax_3d.set_zlabel('Z')
    ax_3d.legend()
    ax_3d.set_title(f'3D Environment (Step {step})')
    
    # Velocities
    velocities = env.v[:n_agents].cpu().numpy()
    speeds = [np.linalg.norm(v) for v in velocities]
    axes[1, 0].bar(range(len(speeds)), speeds)
    axes[1, 0].set_title('Agent Speeds')
    axes[1, 0].set_xlabel('Agent ID')
    axes[1, 0].set_ylabel('Speed (m/s)')
    
    # Distances to obstacles
    vec_to_nearest = env.find_vec_to_nearest_pt()[:n_agents]
    distances = torch.norm(vec_to_nearest, dim=1).cpu().numpy()
    axes[1, 1].bar(range(len(distances)), distances)
    axes[1, 1].set_title('Distance to Nearest Obstacle')
    axes[1, 1].set_xlabel('Agent ID')
    axes[1, 1].set_ylabel('Distance (m)')
    axes[1, 1].axhline(y=env.margin[0].cpu().item(), 
                       color='r', linestyle='--', alpha=0.7, label='Safety Margin')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"Figure saved as {save_fig}")
    
    plt.show()


def barrier_loss(x, v_to_pt):
    """Barrier function for obstacle avoidance."""
    return (v_to_pt * (1 - x).relu().pow(2)).mean()


def train_step(env, model, optim, args):
    """Execute one training step."""
    env.reset()
    model.reset()
    
    # Storage for trajectories
    p_history = []
    v_history = []
    target_v_history = []
    vec_to_pt_history = []
    v_preds = []
    
    h = None
    act_buffer = [env.act]
    ctl_dt = 1 / 15
    
    # Simulation loop
    for t in range(args.timesteps):
        # Render and track
        depth, _ = env.render(ctl_dt)
        p_history.append(env.p)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())
        
        # Update environment
        target_v_raw = env.p_target - env.p.detach()
        env.run(act_buffer[t], ctl_dt, target_v_raw)
        
        # Prepare neural network input
        R = env.R
        fwd = env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)
        
        # Target velocity with speed limits
        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / (target_v_norm + 1e-8)
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
        
        # State vector
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        state = torch.cat([
            local_v,
            torch.squeeze(target_v[:, None] @ R, 1),
            env.R[:, 2],
            env.margin[:, None]
        ], -1)
        
        # Neural network forward pass
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)
        act, _, h = model(x, state, h)
        
        # Process actions
        a_pred, v_pred, *_ = (R @ act.reshape(args.batch_size, 3, -1)).unbind(-1)
        v_preds.append(v_pred)
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
        
        # Store trajectory data
        v_history.append(env.v)
        target_v_history.append(target_v)
    
    # Compute losses
    p_history = torch.stack(p_history)
    v_history = torch.stack(v_history)
    target_v_history = torch.stack(target_v_history)
    vec_to_pt_history = torch.stack(vec_to_pt_history)
    act_buffer = torch.stack(act_buffer)
    v_preds = torch.stack(v_preds)
    
    # Velocity tracking loss
    if len(v_history) > 30:
        v_history_cum = v_history.cumsum(0)
        v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30
        delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
        loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))
    else:
        loss_v = torch.tensor(0.0, device=env.device)
    
    # Other losses
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()
    
    # Obstacle avoidance
    distance = torch.norm(vec_to_pt_history, 2, -1) - env.margin
    if distance.shape[0] > 1:
        with torch.no_grad():
            v_to_pt = (-torch.diff(distance, 1, 0) * 135).clamp_min(1)
        loss_obj_avoidance = barrier_loss(distance[1:], v_to_pt)
        loss_collide = F.softplus(distance[1:].mul(-32)).mul(v_to_pt).mean()
    else:
        loss_obj_avoidance = torch.tensor(0.0, device=env.device)
        loss_collide = torch.tensor(0.0, device=env.device)
    
    # Ground avoidance
    loss_ground = p_history[..., 2].relu().pow(2).mean()
    
    # Combined loss
    loss = (1.0 * loss_v + 
            2.0 * loss_obj_avoidance +
            0.01 * loss_d_acc +
            2.0 * loss_v_pred +
            5.0 * loss_collide +
            0.0 * loss_ground)
    
    # Optimization
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    # Metrics
    with torch.no_grad():
        speed_history = v_history.norm(2, -1)
        success = torch.all(distance.flatten() > 0, 0)
        success_rate = success.sum() / args.batch_size
        
        metrics = {
            'total_loss': float(loss),
            'velocity_loss': float(loss_v),
            'collision_loss': float(loss_collide),
            'obj_avoidance_loss': float(loss_obj_avoidance),
            'success_rate': float(success_rate),
            'avg_speed': float(speed_history.mean()),
            'max_speed': float(speed_history.max())
        }
    
    return metrics


def main():
    """Main training loop."""
    args = parse_arguments()
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = setup_device(args.device)
    env = create_environment(args, device)
    model = create_model(device)
    
    # Optimizer
    optim = AdamW(model.parameters(), args.lr)
    scheduler = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)
    
    # Training loop
    print(f"Starting training for {args.num_iters} iterations...")
    
    metrics_history = defaultdict(list)
    
    try:
        pbar = tqdm(range(args.num_iters), desc="Training")
        
        for i in pbar:
            # Training step
            metrics = train_step(env, model, optim, args)
            scheduler.step()
            
            # Store metrics
            for k, v in metrics.items():
                metrics_history[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.3f}",
                'success': f"{metrics['success_rate']:.2f}",
                'speed': f"{metrics['avg_speed']:.2f}"
            })
            
            # Periodic visualization
            if (i + 1) % 200 == 0:
                print(f"\nStep {i+1}: Loss = {metrics['total_loss']:.4f}, "
                      f"Success = {metrics['success_rate']:.3f}")
                visualize_state(env, step=i+1, save_fig=f"step_{i+1}.png")
        
        print("\nTraining completed!")
        
        # Save model
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved as {args.save_model}")
        
        # Final metrics
        print(f"\nFinal Results:")
        print(f"  Loss: {metrics_history['total_loss'][-1]:.4f}")
        print(f"  Success Rate: {metrics_history['success_rate'][-1]:.3f}")
        print(f"  Average Speed: {metrics_history['avg_speed'][-1]:.2f} m/s")
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(metrics_history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(metrics_history['success_rate'])
        axes[0, 1].set_title('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(metrics_history['collision_loss'], label='Collision')
        axes[1, 0].plot(metrics_history['obj_avoidance_loss'], label='Avoidance')
        axes[1, 0].set_title('Component Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(metrics_history['avg_speed'])
        axes[1, 1].set_title('Average Speed')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Training curves saved as training_curves.png")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        torch.save(model.state_dict(), f"interrupted_{args.save_model}")
        print(f"Model saved as interrupted_{args.save_model}")


if __name__ == "__main__":
    main()