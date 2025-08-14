from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date
import os
from sqlalchemy import or_

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///workorders.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

db = SQLAlchemy(app)

PRIORITIES = ['Low', 'Medium', 'High', 'Urgent']
STATUSES = ['Open', 'In Progress', 'On Hold', 'Completed', 'Canceled']


class WorkOrder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=True)
    priority = db.Column(db.String(20), nullable=False, default='Medium')
    status = db.Column(db.String(20), nullable=False, default='Open')
    requested_by = db.Column(db.String(80), nullable=True)
    assigned_to = db.Column(db.String(80), nullable=True)
    location = db.Column(db.String(120), nullable=True)
    category = db.Column(db.String(80), nullable=True)
    contact_phone = db.Column(db.String(30), nullable=True)
    due_date = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


with app.app_context():
    db.create_all()


@app.route('/')
def home():
    return redirect(url_for('list_workorders'))


@app.route('/workorders')
def list_workorders():
    query = WorkOrder.query
    search = request.args.get('q', '').strip()
    status_filter = request.args.get('status', '').strip()
    priority_filter = request.args.get('priority', '').strip()

    if search:
        like_term = f"%{search}%"
        query = query.filter(
            or_(
                WorkOrder.title.ilike(like_term),
                WorkOrder.description.ilike(like_term),
                WorkOrder.requested_by.ilike(like_term),
                WorkOrder.assigned_to.ilike(like_term),
                WorkOrder.location.ilike(like_term),
                WorkOrder.category.ilike(like_term)
            )
        )
    if status_filter:
        query = query.filter(WorkOrder.status == status_filter)
    if priority_filter:
        query = query.filter(WorkOrder.priority == priority_filter)

    workorders = query.order_by(WorkOrder.created_at.desc()).all()
    return render_template('index.html', workorders=workorders, STATUSES=STATUSES, PRIORITIES=PRIORITIES, search=search, status_filter=status_filter, priority_filter=priority_filter)


@app.route('/workorders/new', methods=['GET'])
def new_workorder():
    return render_template('create_edit.html', workorder=None, STATUSES=STATUSES, PRIORITIES=PRIORITIES, form_action=url_for('create_workorder'), form_title='New Work Order', submit_label='Create')


def parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d').date()
    except ValueError:
        return None


@app.route('/workorders', methods=['POST'])
def create_workorder():
    title = request.form.get('title', '').strip()
    if not title:
        flash('Title is required.', 'error')
        return redirect(url_for('new_workorder'))

    workorder = WorkOrder(
        title=title,
        description=request.form.get('description', '').strip() or None,
        priority=request.form.get('priority') or 'Medium',
        status=request.form.get('status') or 'Open',
        requested_by=request.form.get('requested_by', '').strip() or None,
        assigned_to=request.form.get('assigned_to', '').strip() or None,
        location=request.form.get('location', '').strip() or None,
        category=request.form.get('category', '').strip() or None,
        contact_phone=request.form.get('contact_phone', '').strip() or None,
        due_date=parse_date(request.form.get('due_date'))
    )
    db.session.add(workorder)
    db.session.commit()
    flash('Work order created.', 'success')
    return redirect(url_for('list_workorders'))


@app.route('/workorders/<int:workorder_id>')
def show_workorder(workorder_id):
    workorder = WorkOrder.query.get_or_404(workorder_id)
    return render_template('detail.html', workorder=workorder)


@app.route('/workorders/<int:workorder_id>/edit', methods=['GET'])
def edit_workorder(workorder_id):
    workorder = WorkOrder.query.get_or_404(workorder_id)
    return render_template('create_edit.html', workorder=workorder, STATUSES=STATUSES, PRIORITIES=PRIORITIES, form_action=url_for('update_workorder', workorder_id=workorder.id), form_title='Edit Work Order', submit_label='Save Changes')


@app.route('/workorders/<int:workorder_id>/edit', methods=['POST'])
def update_workorder(workorder_id):
    workorder = WorkOrder.query.get_or_404(workorder_id)
    title = request.form.get('title', '').strip()
    if not title:
        flash('Title is required.', 'error')
        return redirect(url_for('edit_workorder', workorder_id=workorder.id))

    workorder.title = title
    workorder.description = request.form.get('description', '').strip() or None
    workorder.priority = request.form.get('priority') or 'Medium'
    workorder.status = request.form.get('status') or 'Open'
    workorder.requested_by = request.form.get('requested_by', '').strip() or None
    workorder.assigned_to = request.form.get('assigned_to', '').strip() or None
    workorder.location = request.form.get('location', '').strip() or None
    workorder.category = request.form.get('category', '').strip() or None
    workorder.contact_phone = request.form.get('contact_phone', '').strip() or None
    workorder.due_date = parse_date(request.form.get('due_date'))

    db.session.commit()
    flash('Work order updated.', 'success')
    return redirect(url_for('show_workorder', workorder_id=workorder.id))


@app.route('/workorders/<int:workorder_id>/delete', methods=['POST'])
def delete_workorder(workorder_id):
    workorder = WorkOrder.query.get_or_404(workorder_id)
    db.session.delete(workorder)
    db.session.commit()
    flash('Work order deleted.', 'success')
    return redirect(url_for('list_workorders'))


@app.template_filter('datefmt')
def format_date(value, fmt='%Y-%m-%d'):
    if value is None:
        return ''
    if isinstance(value, datetime):
        return value.strftime(fmt)
    if isinstance(value, date):
        return value.strftime(fmt)
    return str(value)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))