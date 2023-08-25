best_ret = None
best_uncertain_type, trace_uncertain_type, ensemble_uncertain_type, no_uncertain_type = -1, 0, 1, 2
min_pull_apart_dist = 1e10
np.random.shuffle(endpoints)
for hulk_endpoint in endpoints:
    ret = network_points(img, cond_grip_pose=hulk_endpoint[0], neck_point=hulk_endpoint[1], vis=vis)
    left_coords, right_coords, trace_uncertain, ensemble_uncertain, pull_apart_dist = ret
    if trace_uncertain and trace_uncertain_type > best_uncertain_type:
        best_ret = ret
        best_uncertain_type = trace_uncertain_type
    elif ensemble_uncertain and ensemble_uncertain_type > best_uncertain_type:
        best_ret = ret
        best_uncertain_type = ensemble_uncertain_type
    elif not ensemble_uncertain and (no_uncertain_type > best_uncertain_type or pull_apart_dist < min_pull_apart_dist):
        best_ret = ret
        best_uncertain_type = no_uncertain_type
        min_pull_apart_dist = pull_apart_dist
if best_ret is not None:
    left_coords, right_coords, trace_uncertain, ensemble_uncertain, pull_apart_dist = best_ret
    left_coords = closest_valid_point(img.color._data, img.depth._data, left_coords) if left_coords is not None else None
    right_coords = closest_valid_point(img.color._data, img.depth._data, right_coords) if left_coords is not None else None
    logger.info(f"trace uncertain {trace_uncertain} and ensemble uncertain {ensemble_uncertain}")
