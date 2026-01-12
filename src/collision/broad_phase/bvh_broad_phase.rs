use core::marker::PhantomData;

use crate::{
    collider_tree::{
        ColliderTree, ColliderTreeProxy, ColliderTreeProxyFlags, ColliderTreeProxyKey,
        ColliderTrees, MovedProxies, ProxyId,
    },
    collision::broad_phase::BroadPhaseDiagnostics,
    data_structures::pair_key::PairKey,
    dynamics::solver::joint_graph::JointGraph,
    prelude::*,
};
use bevy::{
    ecs::system::{StaticSystemParam, SystemParamItem},
    prelude::*,
    tasks::{ComputeTaskPool, ParallelSlice},
};

/// A [broad phase](crate::collision::broad_phase) plugin that uses a [Bounding Volume Hierarchy (BVH)][BVH]
/// to efficiently find pairs of colliders with overlapping AABBs.
///
/// The BVH structures are provided by [`ColliderTrees`].
///
/// [`CollisionHooks`] can be provided with generics to apply custom filtering for collision pairs.
///
/// See the [`broad_phase`](crate::collision::broad_phase) module for more information
/// and an example of creating a custom broad phase plugin.
///
/// [BVH]: https://en.wikipedia.org/wiki/Bounding_volume_hierarchy
pub struct BvhBroadPhasePlugin<H: CollisionHooks = ()>(PhantomData<H>);

impl<H: CollisionHooks> Default for BvhBroadPhasePlugin<H> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<H: CollisionHooks + 'static> Plugin for BvhBroadPhasePlugin<H>
where
    for<'w, 's> SystemParamItem<'w, 's, H>: CollisionHooks,
{
    fn build(&self, app: &mut App) {
        app.add_systems(
            PhysicsSchedule,
            collect_collision_pairs::<H>.in_set(BroadPhaseSystems::CollectCollisions),
        );
    }
}

fn collect_collision_pairs<H: CollisionHooks>(
    trees: ResMut<ColliderTrees>,
    moved_proxies: Res<MovedProxies>,
    collider_of_query: Query<&ColliderOf>,
    hooks: StaticSystemParam<H>,
    par_commands: ParallelCommands,
    mut contact_graph: ResMut<ContactGraph>,
    joint_graph: Res<JointGraph>,
    mut diagnostics: ResMut<BroadPhaseDiagnostics>,
) where
    for<'w, 's> SystemParamItem<'w, 's, H>: CollisionHooks,
{
    let start = crate::utils::Instant::now();

    let hooks = hooks.into_inner();
    let mut broad_collision_pairs = Vec::new();

    // Perform tree queries for all moving proxies.
    let pairs = moved_proxies.proxies().par_splat_map(
        ComputeTaskPool::get(),
        None,
        |_chunk_index, proxies| {
            let mut pairs = Vec::new();

            par_commands.command_scope(|mut commands| {
                for proxy_key1 in proxies {
                    let proxy_id1 = proxy_key1.id();
                    let proxy_type1 = proxy_key1.body();

                    // Get the proxy from its appropriate tree.
                    let tree = trees.tree_for_body(proxy_type1);
                    let proxy1 = tree.get_proxy(proxy_key1.id()).unwrap();

                    // Query dynamic tree.
                    query_tree(
                        &trees.dynamic_tree,
                        RigidBody::Dynamic,
                        *proxy_key1,
                        proxy_id1,
                        proxy_type1,
                        proxy1,
                        &moved_proxies,
                        &hooks,
                        &mut commands,
                        &contact_graph,
                        &joint_graph,
                        &mut pairs,
                    );

                    // Query kinematic tree.
                    query_tree(
                        &trees.kinematic_tree,
                        RigidBody::Kinematic,
                        *proxy_key1,
                        proxy_id1,
                        proxy_type1,
                        proxy1,
                        &moved_proxies,
                        &hooks,
                        &mut commands,
                        &contact_graph,
                        &joint_graph,
                        &mut pairs,
                    );

                    // Skip static-static collisions unless sensors are involved.
                    if proxy1.is_static() && !proxy1.is_sensor() {
                        continue;
                    }

                    // Query static tree.
                    query_tree(
                        &trees.static_tree,
                        RigidBody::Static,
                        *proxy_key1,
                        proxy_id1,
                        proxy_type1,
                        proxy1,
                        &moved_proxies,
                        &hooks,
                        &mut commands,
                        &contact_graph,
                        &joint_graph,
                        &mut pairs,
                    );
                }
            });

            pairs
        },
    );

    // Drain the pairs into a single vector.
    for mut chunk in pairs {
        broad_collision_pairs.append(&mut chunk);
    }

    // TODO: Set flags for events and hooks etc.
    for (entity1, entity2) in broad_collision_pairs {
        let mut contact_edge = ContactEdge::new(entity1, entity2);

        if let (Ok(collider_of1), Ok(collider_of2)) = (
            collider_of_query.get(entity1),
            collider_of_query.get(entity2),
        ) {
            contact_edge.body1 = Some(collider_of1.body);
            contact_edge.body2 = Some(collider_of2.body);
        }

        contact_graph.add_edge_with(contact_edge, |contact_pair| {
            if let (Ok(collider_of1), Ok(collider_of2)) = (
                collider_of_query.get(entity1),
                collider_of_query.get(entity2),
            ) {
                contact_pair.body1 = Some(collider_of1.body);
                contact_pair.body2 = Some(collider_of2.body);
            }
        });
    }

    diagnostics.find_pairs += start.elapsed();
}

#[inline]
fn query_tree(
    tree: &ColliderTree,
    tree_type: RigidBody,
    proxy_key1: ColliderTreeProxyKey,
    proxy_id1: ProxyId,
    proxy_type1: RigidBody,
    proxy1: &ColliderTreeProxy,
    moved_proxies: &MovedProxies,
    hooks: &impl CollisionHooks,
    commands: &mut Commands,
    contact_graph: &ContactGraph,
    joint_graph: &JointGraph,
    pairs: &mut Vec<(Entity, Entity)>,
) {
    tree.bvh.aabb_traverse(proxy1.aabb, |bvh, node_index| {
        let node = bvh.nodes[node_index as usize];
        let start = node.first_index as usize;
        let end = start + node.prim_count as usize;

        for node_primitive_index in start..end {
            let proxy_id2 = ProxyId::new(tree.bvh.primitive_indices[node_primitive_index]);
            let proxy_key2 = ColliderTreeProxyKey::new(proxy_id2, tree_type);

            // Skip self-collision.
            if proxy_key1 == proxy_key2 {
                continue;
            }

            // Avoid duplicate pairs for moving proxies.
            // Most of the time, only dynamic and kinematic bodies will be moving, but static bodies
            // can also be in the move set (ex: when spawned or teleported).
            // TODO: Verify that this logic is correct for all cases.
            //       It might be wrong for static-static sensors.
            let other_greater = ((tree_type as u8) < (proxy_type1 as u8))
                || (tree_type == proxy_type1 && proxy_id2.id() < proxy_id1.id());
            if other_greater && moved_proxies.contains(proxy_key2) {
                // Both proxies are moving, so the other query will handle this pair.
                continue;
            }

            let proxy2 = tree.get_proxy(proxy_id2).unwrap();

            // Check if the layers interact.
            if !proxy1.layers.interacts_with(proxy2.layers) {
                continue;
            }

            // No collisions between colliders on the same body.
            if proxy1.body == proxy2.body {
                continue;
            }

            let entity1 = proxy1.entity;
            let entity2 = proxy2.entity;

            // Avoid duplicate pairs.
            let pair_key = PairKey::new(entity1.index(), entity2.index());
            if contact_graph.contains_key(&pair_key) {
                continue;
            }

            // Check if a joint disables contacts between the two bodies.
            if joint_graph
                .joints_between(proxy1.body, proxy2.body)
                .any(|edge| edge.collision_disabled)
            {
                continue;
            }

            // Apply user-defined filter.
            if proxy1
                .flags
                .union(proxy2.flags)
                .contains(ColliderTreeProxyFlags::CUSTOM_FILTER)
            {
                let should_collide = hooks.filter_pairs(entity1, entity2, commands);
                if !should_collide {
                    continue;
                }
            }

            pairs.push((entity1, entity2));
        }

        true
    });
}
