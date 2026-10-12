"""Orchestration of the genetic algorithm: fitness by simulated matches, training run, artifacts.

Unlike ``agents/`` and ``training/`` this package may drive the game engine (it plays matches), so
the pure GA operators live in :mod:`poker_arena.training.ga` and the poker-specific evaluation here.
"""
