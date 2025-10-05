document.addEventListener('DOMContentLoaded', function() {
    // Initialize all interactive elements
    initializeFavorites();
    initializeFilters();
    initializeSearch();
    initializeNavigation();
    initializeProductCards();
    initializeUserActions();
});

// Favorite functionality
function initializeFavorites() {
    const favoriteButtons = document.querySelectorAll('.favorite-btn, .btn-secondary');
    
    favoriteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Check if user is logged in
            const currentUser = document.querySelector('.nav-value')?.textContent;
            if (!currentUser || currentUser === 'Tài khoản') {
                alert('Vui lòng đăng nhập để sử dụng tính năng yêu thích');
                window.location.href = '/login';
                return;
            }
            
            const carId = this.dataset.carId || this.closest('.product-card')?.dataset.carId;
            if (!carId) {
                console.error('Không tìm thấy ID xe');
                return;
            }
            
            // Send request to server
            fetch('/favorite', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ car_id: carId })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update button appearance
                    if (data.action === 'added') {
                        this.style.background = '#ff6b6b';
                        this.textContent = '❤️';
                        showNotification('Đã thêm vào yêu thích!', 'success');
                    } else {
                        this.style.background = 'rgba(255,255,255,0.1)';
                        this.textContent = '🤍';
                        showNotification('Đã xóa khỏi yêu thích!', 'info');
                    }
                    
                    // Update favorite counter
                    updateFavoriteCounter(data.favorite_count);
                } else {
                    showNotification(data.message || 'Có lỗi xảy ra', 'error');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showNotification('Có lỗi xảy ra khi xử lý yêu thích', 'error');
            });
        });
    });
}

// Filter functionality
function initializeFilters() {
    const filterBtn = document.querySelector('.filter-btn');
    const priceInputs = document.querySelectorAll('.price-input');
    const checkboxes = document.querySelectorAll('.checkbox-group input[type="checkbox"]');
    
    if (filterBtn) {
        filterBtn.addEventListener('click', function(e) {
            e.preventDefault();
            applyFilters();
        });
    }
    
    // Handle "Tất cả" checkbox
    const allCheckbox = document.querySelector('input[value="all"]');
    if (allCheckbox) {
        allCheckbox.addEventListener('change', function() {
            if (this.checked) {
                checkboxes.forEach(checkbox => {
                    if (checkbox !== this) checkbox.checked = false;
                });
            }
        });
    }
    
    // Handle other checkboxes
    checkboxes.forEach(checkbox => {
        if (checkbox.value !== 'all') {
            checkbox.addEventListener('change', function() {
                if (this.checked && allCheckbox) {
                    allCheckbox.checked = false;
                }
            });
        }
    });
}

function applyFilters() {
    const minPrice = document.querySelector('.price-input:first-child')?.value;
    const maxPrice = document.querySelector('.price-input:last-child')?.value;
    const checkedBrands = Array.from(document.querySelectorAll('.checkbox-group input:checked'))
                              .map(cb => cb.value)
                              .filter(val => val !== 'all');
    
    // Create form data
    const formData = new FormData();
    if (minPrice) formData.append('min_price', minPrice);
    if (maxPrice) formData.append('max_price', maxPrice);
    if (checkedBrands.length > 0) {
        formData.append('brand', checkedBrands[0]); // For simplicity, use first selected brand
    }
    
    // Submit filter request
    fetch('/filter', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        // Update page content (in a real app, you'd update specific sections)
        document.querySelector('.product-grid').innerHTML = 
            new DOMParser().parseFromString(html, 'text/html')
                          .querySelector('.product-grid').innerHTML;
        
        // Reinitialize interactions for new content
        initializeFavorites();
        initializeProductCards();
        
        showNotification('Đã áp dụng bộ lọc!', 'success');
    })
    .catch(error => {
        console.error('Filter error:', error);
        showNotification('Có lỗi khi áp dụng bộ lọc', 'error');
    });
}

// Search functionality
function initializeSearch() {
    const searchForm = document.querySelector('.search-form');
    const searchInput = document.querySelector('.search-input');
    const searchBtn = document.querySelector('.search-btn');
    
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            e.preventDefault();
            performSearch();
        });
    }
    
    if (searchBtn) {
        searchBtn.addEventListener('click', function(e) {
            e.preventDefault();
            performSearch();
        });
    }
    
    // Search on Enter key
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                performSearch();
            }
        });
    }
}

function performSearch() {
    const searchInput = document.querySelector('.search-input');
    const searchTerm = searchInput?.value.trim();
    
    if (!searchTerm) {
        showNotification('Vui lòng nhập từ khóa tìm kiếm', 'warning');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('search', searchTerm);
    
    // Submit search request
    fetch('/search', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        // Update page content
        const newDoc = new DOMParser().parseFromString(html, 'text/html');
        const newGrid = newDoc.querySelector('.product-grid');
        
        if (newGrid) {
            document.querySelector('.product-grid').innerHTML = newGrid.innerHTML;
            
            // Reinitialize interactions
            initializeFavorites();
            initializeProductCards();
            
            showNotification(`Tìm thấy kết quả cho: "${searchTerm}"`, 'success');
        }
    })
    .catch(error => {
        console.error('Search error:', error);
        showNotification('Có lỗi khi tìm kiếm', 'error');
    });
}

// Navigation functionality
function initializeNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navItems.forEach(item => {
        item.addEventListener('click', function() {
            const navText = this.querySelector('.nav-label')?.textContent;
            
            if (navText === 'Đăng nhập') {
                window.location.href = '/login';
            } else if (navText === 'Yêu thích') {
                window.location.href = '/favorites';
            } else if (navText === 'Thông báo') {
                showNotification('Tính năng thông báo đang phát triển', 'info');
            }
        });
    });
    
    // Brand navigation
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const brand = this.textContent.trim();
            filterByBrand(brand);
        });
    });
}

function filterByBrand(brand) {
    const formData = new FormData();
    formData.append('brand', brand);
    
    fetch('/filter', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        const newDoc = new DOMParser().parseFromString(html, 'text/html');
        const newGrid = newDoc.querySelector('.product-grid');
        
        if (newGrid) {
            document.querySelector('.product-grid').innerHTML = newGrid.innerHTML;
            initializeFavorites();
            initializeProductCards();
            showNotification(`Hiển thị xe ${brand}`, 'success');
        }
    })
    .catch(error => {
        console.error('Brand filter error:', error);
        showNotification('Có lỗi khi lọc theo hãng xe', 'error');
    });
}

// Product card interactions
function initializeProductCards() {
    const productCards = document.querySelectorAll('.product-card');
    const detailButtons = document.querySelectorAll('.btn-primary');
    
    productCards.forEach((card, index) => {
        // Add car ID for favorites
        if (!card.dataset.carId) {
            card.dataset.carId = (index + 1).toString();
        }
        
        // Hover effects
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-15px)';
            this.style.boxShadow = '0 25px 50px rgba(0,0,0,0.4)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(-10px)';
            this.style.boxShadow = '0 20px 40px rgba(0,0,0,0.3)';
        });
    });
    
    // Detail buttons
    detailButtons.forEach(button => {
        button.addEventListener('click', function() {
            const carName = this.closest('.product-card')
                               .querySelector('.product-name')?.textContent || 'xe này';
            showNotification(`Xem chi tiết ${carName}`, 'info');
            // In real app: window.location.href = `/car-detail/${carId}`;
        });
    });
}

// User actions
function initializeUserActions() {
    const sortBtn = document.querySelector('.sort-btn');
    
    if (sortBtn) {
        sortBtn.addEventListener('click', function() {
            toggleSortDropdown();
        });
    }
}

function toggleSortDropdown() {
    // Simple sort implementation
    const productGrid = document.querySelector('.product-grid');
    const cards = Array.from(productGrid.querySelectorAll('.product-card'));
    
    // Sort by price (extract from price text)
    cards.sort((a, b) => {
        const priceA = parseFloat(a.querySelector('.product-price').textContent.replace(/[^\d.]/g, ''));
        const priceB = parseFloat(b.querySelector('.product-price').textContent.replace(/[^\d.]/g, ''));
        return priceA - priceB;
    });
    
    // Clear and re-add sorted cards
    productGrid.innerHTML = '';
    cards.forEach(card => productGrid.appendChild(card));
    
    showNotification('Đã sắp xếp theo giá tăng dần', 'success');
}

// Utility functions
function updateFavoriteCounter(count) {
    const favoriteCounter = document.querySelector('.nav-item .nav-value');
    if (favoriteCounter) {
        favoriteCounter.textContent = count;
    }
}

function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '15px 20px',
        borderRadius: '8px',
        color: 'white',
        fontWeight: '600',
        zIndex: '9999',
        opacity: '0',
        transform: 'translateX(100%)',
        transition: 'all 0.3s ease',
        maxWidth: '300px',
        wordWrap: 'break-word'
    });
    
    // Set background color based on type
    const colors = {
        success: '#28a745',
        error: '#dc3545', 
        warning: '#ffc107',
        info: '#17a2b8'
    };
    notification.style.background = colors[type] || colors.info;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Animate in
    requestAnimationFrame(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateX(0)';
    });
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Export functions for global access
window.CarStoreJS = {
    showNotification,
    updateFavoriteCounter,
    filterByBrand,
    performSearch
};