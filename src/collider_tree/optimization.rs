use crate::{collision::broad_phase::BroadPhaseDiagnostics, prelude::*};
use bevy::{
    ecs::world::CommandQueue,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on},
};

/// A plugin that optimizes the dynamic [`ColliderTree`] to maintain good query performance.
pub(super) struct ColliderTreeOptimizationPlugin;

impl Plugin for ColliderTreeOptimizationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ColliderTreeOptimization>()
            .init_resource::<OptimizationTask>();

        app.add_systems(
            PhysicsSchedule,
            optimize_trees.in_set(ColliderTreeSystems::BeginOptimize),
        );

        app.add_systems(
            PhysicsSchedule,
            block_on_optimize_trees.in_set(ColliderTreeSystems::EndOptimize),
        );
    }
}

/// Settings for optimizing the dynamic [`ColliderTree`].
#[derive(Resource, Debug, Default, Reflect)]
pub struct ColliderTreeOptimization {
    /// If `true`, tree optimization will be performed in-place with minimal allocations.
    /// This has the downside that the tree will be unavailable for [spatial queries]
    /// during the simulation step while the optimization is ongoing (ex: in [collision hooks]).
    ///
    /// Otherwise, parts of the the tree will be cloned for the optimization,
    /// allowing spatial queries to use the old tree during the simulation step,
    /// but incurring additional memory allocation overhead.
    ///
    /// For optimal performance, set this to `true` if your application
    /// does not perform spatial queries during the simulation step.
    ///
    /// **Default**: `false`
    ///
    /// [spatial queries]: crate::spatial_query
    /// [collision hooks]: crate::collision::hooks
    pub optimize_in_place: bool,

    /// The optimization mode for the collider tree.
    ///
    /// **Default**: [`TreeOptimizationMode::Adaptive`]
    pub optimization_mode: TreeOptimizationMode,
}

/// The optimization mode for a dynamic [`ColliderTree`].
#[derive(Clone, Copy, Debug, Reflect)]
pub enum TreeOptimizationMode {
    /// The tree is optimized by reinserting [moved proxies](`MovedProxies`).
    ///
    /// This is the fastest method when only a small portion of proxies have moved,
    /// but is less effective for large numbers of moved proxies.
    Reinsert,

    /// The tree is optimized by performing a partial rebuild that only rebuilds
    /// parts of the tree affected by [moved proxies](`MovedProxies`).
    ///
    /// This method is more effective than reinsertion when a moderate number of proxies
    /// have moved. However, if a large portion of proxies have moved, a full rebuild
    /// can be more effective and have less overhead.
    PartialRebuild,

    /// The tree is optimized by performing a full rebuild.
    ///
    /// This method can produce the highest quality tree, and can have less overhead
    /// than other methods when a large portion of proxies have moved.
    /// This makes it suitable for highly dynamic scenes.
    FullRebuild,

    /// The tree is optimized adaptively based on how many proxies have moved.
    ///
    /// - If the ratio of moved proxies to total proxies is below
    ///   `reinsert_threshold`, [`Reinsert`](TreeOptimizationMode::Reinsert) is used.
    /// - If the ratio is between `reinsert_threshold` and `partial_rebuild_threshold`,
    ///   [`PartialRebuild`](TreeOptimizationMode::PartialRebuild) is used.
    /// - Otherwise, [`FullRebuild`](TreeOptimizationMode::FullRebuild) is used.
    ///
    /// This is the default mode.
    Adaptive {
        /// The threshold ratio of moved proxies to total proxies
        /// below which reinsertion is performed.
        ///
        /// **Default**: `0.15`
        reinsert_threshold: f32,

        /// The threshold ratio of moved proxies to total proxies
        /// below which a partial rebuild is performed.
        ///
        /// **Default**: `0.45`
        partial_rebuild_threshold: f32,
    },
}

impl Default for TreeOptimizationMode {
    fn default() -> Self {
        TreeOptimizationMode::Adaptive {
            reinsert_threshold: 0.15,
            partial_rebuild_threshold: 0.45,
        }
    }
}

impl TreeOptimizationMode {
    /// Resolves the optimization mode based on the ratio of moved proxies.
    ///
    /// `moved_ratio` is the ratio of moved proxies to total proxies in the tree.
    #[inline]
    pub fn resolve(&self, moved_ratio: f32) -> TreeOptimizationMode {
        match self {
            TreeOptimizationMode::Adaptive {
                reinsert_threshold,
                partial_rebuild_threshold,
            } => {
                if moved_ratio < *reinsert_threshold {
                    TreeOptimizationMode::Reinsert
                } else if moved_ratio < *partial_rebuild_threshold {
                    TreeOptimizationMode::PartialRebuild
                } else {
                    TreeOptimizationMode::FullRebuild
                }
            }
            other => *other,
        }
    }
}

/// A resource tracking the ongoing optimization task for the dynamic [`ColliderTree`].
#[derive(Resource, Default)]
struct OptimizationTask {
    /// The collider tree being optimized.
    ///
    /// This is taken from the dynamic tree before optimization begins,
    /// and the optimized BVH is written back to the dynamic tree
    /// when the optimization is complete.
    tree: ColliderTree,

    /// The async task performing the rebuild.
    task: Option<Task<CommandQueue>>,
}

/// Begins optimizing the dynamic [`ColliderTree`] to maintain good query performance.
///
/// This spawns an async task that runs concurrently with the simulation step.
fn optimize_trees(
    mut collider_trees: ResMut<ColliderTrees>,
    mut optimization: ResMut<OptimizationTask>,
    optimization_settings: Res<ColliderTreeOptimization>,
    moved_proxies: ResMut<MovedProxies>,
    mut diagnostics: ResMut<BroadPhaseDiagnostics>,
) {
    let start = crate::utils::Instant::now();

    let task_pool = AsyncComputeTaskPool::get();

    // Use the dynamic tree's workspace for the optimization.
    core::mem::swap(
        &mut collider_trees.dynamic_tree.workspace,
        &mut optimization.tree.workspace,
    );

    let moved_ratio =
        moved_proxies.proxies().len() as f32 / collider_trees.dynamic_tree.proxies.len() as f32;

    let mut tree = core::mem::take(&mut optimization.tree);

    if optimization_settings.optimize_in_place {
        core::mem::swap(&mut tree.bvh, &mut collider_trees.dynamic_tree.bvh);
    } else {
        // TODO: Can we avoid cloning the entire BVH?
        tree.bvh.clone_from(&collider_trees.dynamic_tree.bvh);
    }

    let task = match optimization_settings.optimization_mode.resolve(moved_ratio) {
        TreeOptimizationMode::Reinsert => {
            let moved_leaves = moved_proxies
                .proxies()
                .iter()
                .map(|&i| tree.bvh.primitives_to_nodes[i as usize])
                .collect::<Vec<u32>>();

            spawn_optimization_task(task_pool, tree, move |tree| {
                tree.optimize_candidates(&moved_leaves, 1);
            })
        }
        TreeOptimizationMode::PartialRebuild => {
            let moved_leaves = moved_proxies
                .proxies()
                .iter()
                .map(|&i| tree.bvh.primitives_to_nodes[i as usize])
                .collect::<Vec<u32>>();

            spawn_optimization_task(task_pool, tree, move |tree| {
                tree.rebuild_partial(&moved_leaves);
            })
        }
        TreeOptimizationMode::FullRebuild => spawn_optimization_task(task_pool, tree, |tree| {
            tree.rebuild_full();
        }),

        TreeOptimizationMode::Adaptive { .. } => unreachable!(""),
    };

    optimization.task = Some(task);

    diagnostics.optimize += start.elapsed();
}

/// Spawns and returns an async task to optimize the given collider tree
/// using the provided optimization function.
fn spawn_optimization_task(
    task_pool: &AsyncComputeTaskPool,
    mut tree: ColliderTree,
    optimize: impl FnOnce(&mut ColliderTree) + Send + 'static,
) -> Task<CommandQueue> {
    task_pool.spawn(async move {
        optimize(&mut tree);

        let mut command_queue = CommandQueue::default();
        command_queue.push(move |world: &mut World| {
            let mut collider_trees = world
                .get_resource_mut::<ColliderTrees>()
                .expect("ColliderTrees resource missing");
            collider_trees.dynamic_tree.bvh = tree.bvh;
        });
        command_queue
    })
}

/// Completes the optimization of the dynamic [`ColliderTree`] started in [`optimize_trees`].
fn block_on_optimize_trees(
    mut commands: Commands,
    mut collider_trees: ResMut<ColliderTrees>,
    mut optimization: ResMut<OptimizationTask>,
    mut diagnostics: ResMut<BroadPhaseDiagnostics>,
) {
    let start = crate::utils::Instant::now();

    if let Some(task) = &mut optimization.task {
        let mut command_queue = block_on(task);
        commands.append(&mut command_queue);
    }

    optimization.task = None;

    // Restore the dynamic tree's workspace.
    core::mem::swap(
        &mut collider_trees.dynamic_tree.workspace,
        &mut optimization.tree.workspace,
    );

    diagnostics.optimize += start.elapsed();
}
