// static/js/car_store.js - Quản lý chức năng xe hơi
class CarStore {
    constructor() {
        this.currentUser = null;
        this.init();
    }

    init() {
        this.checkAuthentication();
        this.setupEventListeners();
    }

    // Kiểm tra đăng nhập
    checkAuthentication() {
        // Kiểm tra từ session Flask (sẽ được render trong template)
        const userLoggedIn = document.body.getAttribute('data-user-logged-in') === 'true';
        if (userLoggedIn) {
            this.currentUser = {
                email: document.body.getAttribute('data-user-email'),
                name: document.body.getAttribute('data-user-name'),
                role: document.body.getAttribute('data-user-role')
            };
            this.updateUIForLoggedInUser();
        }
    }

    // Cập nhật UI khi đăng nhập
    updateUIForLoggedInUser() {
        // Ẩn nút login, hiện user menu
        const loginBtn = document.querySelector('.lcb-login-button');
        const userMenu = document.getElementById('user-menu-btn');

        if (loginBtn) loginBtn.style.display = 'none';
        if (userMenu) {
            userMenu.innerHTML = `👤 ${this.currentUser.name}`;
            userMenu.style.display = 'block';
        }

        // Thêm dropdown menu nếu chưa có
        this.createUserDropdown();
    }

    // Tạo dropdown menu user
    createUserDropdown() {
        const userBtn = document.getElementById('user-menu-btn');
        if (!userBtn || document.getElementById('user-dropdown')) return;

        const dropdown = document.createElement('div');
        dropdown.id = 'user-dropdown';
        dropdown.style.cssText = `
            display: none;
            position: absolute;
            top: 100%;
            right: 0;
            background: white;
            border-radius: 8px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            min-width: 200px;
            z-index: 1000;
            margin-top: 5px;
        `;

        dropdown.innerHTML = `
            <a href="/profile" class="dropdown-item" style="display: block; padding: 12px 20px; color: #333; text-decoration: none; border-bottom: 1px solid #eee;">👤 Hồ sơ</a>
            <a href="/my_test_drives" class="dropdown-item" style="display: block; padding: 12px 20px; color: #333; text-decoration: none; border-bottom: 1px solid #eee;">🚗 Lịch lái thử</a>
            ${this.currentUser.role === 'admin' ?
            '<a href="/admin" class="dropdown-item" style="display: block; padding: 12px 20px; color: #667eea; text-decoration: none; border-bottom: 1px solid #eee;">⚙️ Quản trị</a>' : ''}
            <a href="/logout" class="dropdown-item" style="display: block; padding: 12px 20px; color: #e74c3c; text-decoration: none;">🚪 Đăng xuất</a>
        `;

        userBtn.parentNode.appendChild(dropdown);

        // Toggle dropdown
        userBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            dropdown.style.display = dropdown.style.display === 'none' ? 'block' : 'none';
        });

        // Đóng khi click bên ngoài
        document.addEventListener('click', () => {
            dropdown.style.display = 'none';
        });
    }

    // Đặt lịch lái thử
    scheduleTestDrive(carId, carName) {
        if (!this.currentUser) {
            alert('Vui lòng đăng nhập để đặt lịch lái thử!');
            window.location.href = '/login';
            return;
        }

        const date = prompt(`Nhập ngày bạn muốn lái thử ${carName} (YYYY-MM-DD):`);
        if (!date) return;

        if (!this.isValidDate(date)) {
            alert('Vui lòng nhập ngày hợp lệ (YYYY-MM-DD)');
            return;
        }

        fetch('/api/test_drive/schedule', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                car_id: carId,
                car_name: carName,
                scheduled_date: date
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('✅ Đã đặt lịch lái thử thành công!');
            } else {
                alert('❌ Lỗi: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('❌ Có lỗi xảy ra khi đặt lịch');
        });
    }

    isValidDate(dateString) {
        const regex = /^\d{4}-\d{2}-\d{2}$/;
        if (!regex.test(dateString)) return false;

        const date = new Date(dateString);
        return date instanceof Date && !isNaN(date);
    }

    setupEventListeners() {
        // Thêm nút "Đặt lái thử" vào các card xe
        this.addTestDriveButtons();
    }

    addTestDriveButtons() {
        // Tìm tất cả card xe và thêm nút lái thử
        const carCards = document.querySelectorAll('.car-card');
        carCards.forEach(card => {
            const carId = card.getAttribute('data-car-id');
            const carName = card.querySelector('.car-model')?.textContent;

            if (carId && carName) {
                const actionsDiv = card.querySelector('.car-actions') || this.createActionsDiv(card);
                const testDriveBtn = document.createElement('button');

                testDriveBtn.innerHTML = '🚗 Đặt lái thử';
                testDriveBtn.style.cssText = `
                    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 14px;
                    margin-top: 8px;
                    width: 100%;
                `;

                testDriveBtn.onclick = () => this.scheduleTestDrive(carId, carName);
                actionsDiv.appendChild(testDriveBtn);
            }
        });
    }

    createActionsDiv(card) {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'car-actions';
        actionsDiv.style.cssText = 'margin-top: 15px; display: flex; flex-direction: column; gap: 8px;';

        const body = card.querySelector('.car-card-body');
        if (body) {
            body.appendChild(actionsDiv);
        }

        return actionsDiv;
    }
}

// Khởi tạo khi trang load
document.addEventListener('DOMContentLoaded', function() {
    window.carStore = new CarStore();
});