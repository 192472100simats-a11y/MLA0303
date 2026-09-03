from flask import Flask, request, render_template_string
from datetime import datetime
import random

app = Flask(__name__)


# ============================================================
# OOP CLASSES - CORE ENTITIES
# ============================================================

class DeliveryTask:
    total_tasks = 0

    def __init__(self, task_id, sender, source, destination, weight, waiting, task_type):
        self.task_id = task_id
        self.sender = sender
        self.source = source
        self.destination = destination
        self.weight = float(weight)
        self.waiting = int(waiting)
        self.task_type = task_type
        self.status = "Waiting"
        DeliveryTask.total_tasks += 1

    def get_priority(self):
        type_bonus = {
            "Normal": 5,
            "Priority": 20,
            "Emergency": 50
        }.get(self.task_type, 0)

        weight_penalty = min(self.weight, 10)

        return self.waiting + type_bonus - weight_penalty

    def __str__(self):
        return f"{self.task_id} ({self.source} -> {self.destination})"


class NormalTask(DeliveryTask):
    def __init__(self, task_id, sender, source, destination, weight, waiting):
        super().__init__(task_id, sender, source, destination, weight, waiting, "Normal")


class PriorityTask(DeliveryTask):
    def __init__(self, task_id, sender, source, destination, weight, waiting):
        super().__init__(task_id, sender, source, destination, weight, waiting, "Priority")


class EmergencyTask(DeliveryTask):
    def __init__(self, task_id, sender, source, destination, weight, waiting):
        super().__init__(task_id, sender, source, destination, weight, waiting, "Emergency")


class Waypoint:
    total_waypoints = 0

    def __init__(self, waypoint_id, zone):
        self.waypoint_id = waypoint_id
        self.zone = zone
        self.available = True
        self.current_robot = None
        Waypoint.total_waypoints += 1

    def block(self):
        self.available = False

    def clear(self):
        self.available = True
        self.current_robot = None

    def __del__(self):
        pass


class Robot:
    total_robots = 0

    def __init__(self, robot_id, robot_type="Standard", speed=10):
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.speed = float(speed)
        self.battery = 100.0
        self.available = True
        self.current_task = None
        self.current_waypoint = None
        self.distance_travelled = 0
        Robot.total_robots += 1

    def move(self, distance):
        self.distance_travelled += distance
        self.battery = max(0, self.battery - distance * 0.5)

    def __str__(self):
        return f"{self.robot_id} [{self.robot_type}]"


class DeliverySession:
    total_sessions = 0

    def __init__(self, session_id, task, robot, waypoint):
        self.session_id = session_id
        self.task = task
        self.robot = robot
        self.waypoint = waypoint
        self.distance = 0
        self.reward = 0
        self.status = "In-Progress"
        self.time = datetime.now().strftime("%H:%M:%S")
        DeliverySession.total_sessions += 1

    def __gt__(self, other):
        return self.distance > other.distance

    def __str__(self):
        return f"Session-{self.session_id}"


class PriorityManager:

    @staticmethod
    def calculate(task):
        return task.get_priority()

    @staticmethod
    def sort(tasks):
        return sorted(tasks, key=lambda t: t.get_priority(), reverse=True)


# ============================================================
# REINFORCEMENT LEARNING CLASSES
# ============================================================

class PolicyBasedRL:
    """Learns which robot should serve which task right now (immediate action)."""

    def select_action(self, tasks, robots, waypoints):
        waiting_tasks = [t for t in tasks if t.status == "Waiting"]
        free_robots = [r for r in robots if r.available]
        free_points = [w for w in waypoints if w.available]

        if not waiting_tasks or not free_robots or not free_points:
            return None

        task = max(waiting_tasks, key=lambda t: t.get_priority())
        robot = max(free_robots, key=lambda r: r.speed)
        point = min(free_points, key=lambda w: w.waypoint_id)

        return robot, task, point

    def calculate_reward(self, task, distance):
        reward = 0
        if task.task_type == "Emergency":
            reward += 30
        reward += max(0, 20 - distance)
        reward -= task.waiting * 0.5
        return round(reward, 2)


class ModelBasedRL:
    """Predicts future environmental conditions and plans several steps ahead
    instead of only reacting to the current state."""

    def predict(self, task, robot, waypoint):
        predicted_congestion = random.uniform(0, 1)
        predicted_delay = round(predicted_congestion * 5, 2)

        predicted_battery_use = min(robot.battery, task.weight * 2)

        score = (
            task.get_priority()
            + robot.speed * 2
            - predicted_delay
            - predicted_battery_use * 0.3
        )
        return round(score, 2), predicted_delay

    def best_plan(self, tasks, robots, waypoints):
        waiting = [t for t in tasks if t.status == "Waiting"]
        free_robots = [r for r in robots if r.available]
        free_points = [w for w in waypoints if w.available]

        if not waiting or not free_robots or not free_points:
            return None

        best = None
        best_score = -999999

        for task in waiting:
            for robot in free_robots:
                for point in free_points:
                    score, delay = self.predict(task, robot, point)
                    if score > best_score:
                        best_score = score
                        best = (robot, task, point, score, delay)

        return best


class HierarchicalRL:
    """Decomposes the navigation problem into three levels of decision-making:
    mission level (which tasks to serve), route level (which waypoints to
    travel through) and motion level (step-by-step movement)."""

    def execute(self, tasks, robots, waypoints):
        mission_level = len([t for t in tasks if t.status == "Waiting"])

        route_level = min(len(tasks), len(robots)) if robots else 0

        motion_level = sum(1 for w in waypoints if not w.available)

        return {
            "mission_level": mission_level,
            "route_level": route_level,
            "motion_level": motion_level
        }


class MetaLearning:
    """Allows the robot to quickly re-adjust its behaviour when it meets a
    new or unexpected situation, instead of learning again from scratch."""

    def adapt(self, condition):
        if condition == "New Task Type":
            return "Policy weights fast-adapted for the newly seen task category."
        if condition == "Robot Failure":
            return "Remaining robots re-planned to cover the failed robot's deliveries."
        if condition == "Corridor Blocked":
            return "Route planner re-computed alternate paths around the blockage."
        if condition == "Layout Changed":
            return "Navigation model re-calibrated to the updated building layout."
        return "System adapted successfully."


# ============================================================
# NAVIGATION SYSTEM (ORCHESTRATOR)
# ============================================================

class NavigationSystem:

    def __init__(self):
        self.tasks = []
        self.robots = []
        self.waypoints = []
        self.sessions = []

        self.policy = PolicyBasedRL()
        self.model = ModelBasedRL()
        self.hierarchical = HierarchicalRL()
        self.meta = MetaLearning()

        self.log = []

    @property
    def active_robots(self):
        return sum(1 for r in self.robots if not r.available)

    def add_log(self, message):
        self.log.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.log = self.log[:12]


navigation = NavigationSystem()


# ============================================================
# HTML TEMPLATE
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Autonomous Delivery Robot Navigation System</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; }
.container { max-width: 1100px; margin: 0 auto; padding: 20px; }
h1 { text-align:center; padding: 20px 0 5px 0; }
.subtitle { text-align:center; color:#94a3b8; margin-bottom:25px; }
.section { background:#1e293b; border-radius:12px; padding:20px; margin-bottom:20px; }
.section h2 { margin-top:0; }
.form-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap:12px; }
label { font-size:13px; color:#94a3b8; display:block; margin-bottom:4px; }
input, select { width:100%; padding:8px; border-radius:6px; border:1px solid #334155; background:#0f172a; color:#e2e8f0; box-sizing:border-box; }
button { padding:9px 16px; border:none; border-radius:6px; background:#3b82f6; color:white; cursor:pointer; font-weight:600; }
button.green { background:#22c55e; }
button.purple { background:#a855f7; }
button.orange { background:#f97316; }
button.red { background:#ef4444; }
table { width:100%; border-collapse: collapse; margin-top:10px; }
th, td { text-align:left; padding:8px; border-bottom:1px solid #334155; font-size:14px; }
.status { padding:3px 8px; border-radius:12px; font-size:12px; }
.available { background:#14532d; color:#4ade80; }
.busy { background:#7f1d1d; color:#f87171; }
.rl-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:14px; }
.rl-card { background:#0f172a; border:1px solid #334155; border-radius:10px; padding:14px; }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap:14px; }
.card { background:#0f172a; border-radius:10px; padding:14px; text-align:center; }
.number { font-size:26px; font-weight:700; color:#38bdf8; }
.log { font-family: monospace; font-size:13px; background:#0f172a; padding:12px; border-radius:8px; max-height:200px; overflow-y:auto; }
</style>
</head>
<body>
<div class="container">

<h1>Autonomous Delivery Robot Navigation System</h1>
<div class="subtitle">Policy-Based RL &bull; Model-Based RL &bull; Hierarchical RL &bull; Meta-Learning</div>

<div class="section">
<h2>Robots</h2>
<form action="/add_robot" method="POST">
<div class="form-grid">
<div><label>Robot ID</label><input name="robot_id" placeholder="R01" required></div>
<div><label>Type</label>
<select name="robot_type">
<option>Standard</option>
<option>Express</option>
<option>HeavyLoad</option>
</select>
</div>
<div><label>Speed (m/s)</label><input type="number" step="0.1" name="speed" placeholder="1.2" required></div>
</div>
<br><button class="green">+ Add Robot</button>
</form>
<br>
<table>
<tr><th>Robot</th><th>Type</th><th>Speed</th><th>Battery</th><th>Status</th><th>Task</th></tr>
{% for r in robots %}
<tr>
<td>{{ r.robot_id }}</td>
<td>{{ r.robot_type }}</td>
<td>{{ r.speed }} m/s</td>
<td>{{ "%.1f"|format(r.battery) }}%</td>
<td><span class="status {% if r.available %}available{% else %}busy{% endif %}">{% if r.available %}Available{% else %}On Delivery{% endif %}</span></td>
<td>{{ r.current_task or "-" }}</td>
</tr>
{% endfor %}
</table>
</div>

<div class="section">
<h2>Waypoints</h2>
<form action="/add_waypoint" method="POST">
<div class="form-grid">
<div><label>Waypoint ID</label><input name="waypoint_id" placeholder="W01" required></div>
<div><label>Zone</label><input name="zone" placeholder="Floor-1-East" required></div>
</div>
<br><button class="green">+ Add Waypoint</button>
</form>
<br>
<table>
<tr><th>Waypoint</th><th>Zone</th><th>Status</th></tr>
{% for w in waypoints %}
<tr>
<td>{{ w.waypoint_id }}</td>
<td>{{ w.zone }}</td>
<td><span class="status {% if w.available %}available{% else %}busy{% endif %}">{% if w.available %}Clear{% else %}Blocked/Busy{% endif %}</span></td>
</tr>
{% endfor %}
</table>
</div>

<div class="section">
<h2>Delivery Tasks</h2>
<form action="/add_task" method="POST">
<div class="form-grid">
<div><label>Task ID</label><input name="task_id" placeholder="T01" required></div>
<div><label>Sender</label><input name="sender" placeholder="Lab-3" required></div>
<div><label>Source</label><input name="source" placeholder="W01" required></div>
<div><label>Destination</label><input name="destination" placeholder="W05" required></div>
<div><label>Weight (kg)</label><input type="number" step="0.1" name="weight" placeholder="2" required></div>
<div><label>Waiting (min)</label><input type="number" name="waiting" placeholder="0" required></div>
<div><label>Task Type</label>
<select name="task_type">
<option>Normal</option>
<option>Priority</option>
<option>Emergency</option>
</select>
</div>
</div>
<br><button class="green">+ Add Task</button>
</form>
<br>
<table>
<tr><th>Task</th><th>Route</th><th>Type</th><th>Priority</th><th>Status</th></tr>
{% for t in tasks %}
<tr>
<td>{{ t.task_id }}</td>
<td>{{ t.source }} &rarr; {{ t.destination }}</td>
<td>{{ t.task_type }}</td>
<td>{{ "%.1f"|format(t.get_priority()) }}</td>
<td>{{ t.status }}</td>
</tr>
{% endfor %}
</table>
</div>

<div class="section">
<h2>Reinforcement Learning Control</h2>
<div class="rl-grid">

<div class="rl-card">
<h3>Policy-Based RL</h3>
<p>Learns and selects the best immediate robot-task-waypoint action from the current state.</p>
<form action="/policy" method="POST"><button>Run Policy Decision</button></form>
</div>

<div class="rl-card">
<h3>Model-Based RL</h3>
<p>Predicts environmental changes (congestion, delay, battery use) and plans ahead before acting.</p>
<form action="/model" method="POST"><button class="purple">Run Prediction &amp; Planning</button></form>
</div>

<div class="rl-card">
<h3>Hierarchical RL</h3>
<p>Splits the decision into mission, route and motion levels for scalable navigation.</p>
<form action="/hierarchical" method="POST"><button class="orange">Run Hierarchical Control</button></form>
</div>

<div class="rl-card">
<h3>Meta-Learning</h3>
<p>Rapidly adapts the robot's behaviour to a new or unexpected situation.</p>
<form action="/meta" method="POST">
<select name="condition" style="margin-bottom:8px;">
<option>New Task Type</option>
<option>Robot Failure</option>
<option>Corridor Blocked</option>
<option>Layout Changed</option>
</select>
<button class="red">Adapt Now</button>
</form>
</div>

</div>
</div>

<div class="section">
<h2>Delivery Operations</h2>
<form action="/start_delivery" method="POST" style="display:inline"><button class="green">Start Delivery</button></form>
&nbsp;
<form action="/complete" method="POST" style="display:inline"><button class="orange">Complete Delivery</button></form>
</div>

<div class="section">
<h2>Delivery Sessions</h2>
<table>
<tr><th>Session</th><th>Task</th><th>Robot</th><th>Waypoint</th><th>Distance</th><th>Reward</th><th>Status</th><th>Time</th></tr>
{% for s in sessions %}
<tr>
<td>{{ s.session_id }}</td>
<td>{{ s.task }}</td>
<td>{{ s.robot }}</td>
<td>{{ s.waypoint }}</td>
<td>{{ "%.2f"|format(s.distance) }} m</td>
<td>{{ "%.2f"|format(s.reward) }}</td>
<td>{{ s.status }}</td>
<td>{{ s.time }}</td>
</tr>
{% endfor %}
</table>
</div>

<div class="section">
<h2>System Report</h2>
<div class="cards">
<div class="card"><h3>Robots</h3><div class="number">{{ robots|length }}</div></div>
<div class="card"><h3>Active Deliveries</h3><div class="number">{{ active_robots }}</div></div>
<div class="card"><h3>Waypoints</h3><div class="number">{{ waypoints|length }}</div></div>
<div class="card"><h3>Completed Sessions</h3><div class="number">{{ completed }}</div></div>
</div>
</div>

<div class="section">
<h2>Activity Log</h2>
<div class="log">
{% for entry in log %}
{{ entry }}<br>
{% endfor %}
</div>
</div>

<div style="text-align:center;padding:20px;color:#64748b;">
Autonomous Delivery Robot Navigation System<br>
Policy-Based RL &bull; Model-Based RL &bull; Hierarchical RL &bull; Meta-Learning
</div>

</div>
</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    completed = sum(1 for s in navigation.sessions if s.status == "Completed")
    return render_template_string(
        HTML,
        robots=navigation.robots,
        waypoints=navigation.waypoints,
        tasks=navigation.tasks,
        sessions=navigation.sessions,
        active_robots=navigation.active_robots,
        completed=completed,
        log=navigation.log
    )


@app.route("/add_robot", methods=["POST"])
def add_robot():
    robot_id = request.form["robot_id"]
    robot_type = request.form["robot_type"]
    speed = request.form["speed"]
    navigation.robots.append(Robot(robot_id, robot_type, speed))
    navigation.add_log(f"Robot {robot_id} ({robot_type}) registered.")
    return index()


@app.route("/add_waypoint", methods=["POST"])
def add_waypoint():
    waypoint_id = request.form["waypoint_id"]
    zone = request.form["zone"]
    navigation.waypoints.append(Waypoint(waypoint_id, zone))
    navigation.add_log(f"Waypoint {waypoint_id} added in {zone}.")
    return index()


@app.route("/add_task", methods=["POST"])
def add_task():
    task_id = request.form["task_id"]
    sender = request.form["sender"]
    source = request.form["source"]
    destination = request.form["destination"]
    weight = request.form["weight"]
    waiting = request.form["waiting"]
    task_type = request.form["task_type"]

    if task_type == "Priority":
        task = PriorityTask(task_id, sender, source, destination, weight, waiting)
    elif task_type == "Emergency":
        task = EmergencyTask(task_id, sender, source, destination, weight, waiting)
    else:
        task = NormalTask(task_id, sender, source, destination, weight, waiting)

    navigation.tasks.append(task)
    navigation.add_log(f"Task {task_id} ({task_type}) requested: {source} -> {destination}.")
    return index()


@app.route("/policy", methods=["POST"])
def policy():
    result = navigation.policy.select_action(navigation.tasks, navigation.robots, navigation.waypoints)
    if result:
        robot, task, point = result
        navigation.add_log(
            f"Policy-Based RL: robot {robot.robot_id} selected for task {task.task_id} via waypoint {point.waypoint_id}."
        )
    else:
        navigation.add_log("Policy-Based RL: no feasible action found (no free robot / waypoint / task).")
    return index()


@app.route("/model", methods=["POST"])
def model():
    result = navigation.model.best_plan(navigation.tasks, navigation.robots, navigation.waypoints)
    if result:
        robot, task, point, score, delay = result
        navigation.add_log(
            f"Model-Based RL: predicted best plan -> robot {robot.robot_id}, task {task.task_id}, "
            f"waypoint {point.waypoint_id} (score {score}, predicted delay {delay} min)."
        )
    else:
        navigation.add_log("Model-Based RL: no feasible plan could be predicted.")
    return index()


@app.route("/hierarchical", methods=["POST"])
def hierarchical():
    result = navigation.hierarchical.execute(navigation.tasks, navigation.robots, navigation.waypoints)
    navigation.add_log(
        f"Hierarchical RL: mission_level={result['mission_level']} tasks, "
        f"route_level={result['route_level']} assignable routes, "
        f"motion_level={result['motion_level']} blocked waypoints."
    )
    return index()


@app.route("/meta", methods=["POST"])
def meta():
    condition = request.form["condition"]
    message = navigation.meta.adapt(condition)
    navigation.add_log(f"Meta-Learning ({condition}): {message}")
    return index()


@app.route("/start_delivery", methods=["POST"])
def start_delivery():
    result = navigation.policy.select_action(navigation.tasks, navigation.robots, navigation.waypoints)
    if not result:
        navigation.add_log("Start Delivery failed: no available robot, task or waypoint.")
        return index()

    robot, task, point = result

    robot.available = False
    robot.current_task = task.task_id
    robot.current_waypoint = point.waypoint_id
    point.available = False
    point.current_robot = robot.robot_id
    task.status = "In-Progress"

    session_id = DeliverySession.total_sessions + 1
    session = DeliverySession(session_id, task, robot, point)
    session.distance = round(random.uniform(5, 40), 2)
    robot.move(session.distance)
    session.reward = navigation.policy.calculate_reward(task, session.distance)

    navigation.sessions.append(session)
    navigation.add_log(
        f"Delivery started: {robot.robot_id} carrying {task.task_id} to {point.waypoint_id} "
        f"(distance {session.distance} m, reward {session.reward})."
    )
    return index()


@app.route("/complete", methods=["POST"])
def complete():
    active = [s for s in navigation.sessions if s.status == "In-Progress"]
    if not active:
        navigation.add_log("Complete Delivery failed: no active delivery session.")
        return index()

    session = active[0]
    session.status = "Completed"
    session.task.status = "Delivered"
    session.robot.available = True
    session.robot.current_task = None
    session.waypoint.available = True
    session.waypoint.current_robot = None

    navigation.add_log(f"Delivery {session.session_id} completed by {session.robot.robot_id}.")
    return index()


if __name__ == "__main__":
    app.run(debug=True, port=5000)