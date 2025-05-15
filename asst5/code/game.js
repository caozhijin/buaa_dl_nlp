class Game {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 800;
        this.canvas.height = 400;
        
        // 游戏状态
        this.isPlaying = false;
        this.score = 0;
        this.highScore = localStorage.getItem('highScore') || 0;
        
        // 玩家属性
        this.player = {
            x: 50,
            y: this.canvas.height - 40,
            radius: 20,
            velocity: 0,
            gravity: 0.8,
            isJumping: false,
            jumpStartTime: 0,
            maxJumpTime: 200,    // 最大跳跃蓄力时间（毫秒）
            baseJumpForce: -12,  // 基础跳跃力
            maxExtraForce: -10,  // 最大额外跳跃力
            baseColor: '#e74c3c',  // 玩家基础颜色（红色）
            jumpingColor: '#c0392b',  // 玩家跳跃时的颜色（深红色）
            jumpCount: 0,  // 跳跃次数计数
            maxJumps: 2,   // 最大跳跃次数
            doubleJumpMultiplier: 0.7  // 二段跳高度倍率
        };
        
        // 基础障碍物配置
        this.baseObstacleTypes = {
            small: { width: 30, height: 60, color: '#2ecc71' },    // 小型障碍物
            medium: { width: 40, height: 80, color: '#27ae60' },   // 中型障碍物
            large: { width: 50, height: 100, color: '#145a32' }    // 大型障碍物
        };

        // 组合障碍物配置
        this.obstacleTypes = [
            // 单个障碍物
            { type: 'single', pattern: ['small'] },
            { type: 'single', pattern: ['medium'] },
            { type: 'single', pattern: ['large'] },
            // 组合障碍物
            { type: 'combo', pattern: ['small', 'medium'] },           // 小+中
            { type: 'combo', pattern: ['medium', 'small'] },           // 中+小
            { type: 'combo', pattern: ['small', 'medium', 'small'] }   // 小+中+小
        ];
        
        // 障碍物数组
        this.obstacles = [];
        this.obstacleTimer = 0;
        this.minObstacleInterval = 1000;  // 最小间隔
        this.maxObstacleInterval = 2500;  // 最大间隔
        this.nextObstacleInterval = this.getRandomInterval();
        
        // 游戏速度
        this.gameSpeed = 5;
        
        // 绑定事件处理器
        this.bindEvents();
        
        // 初始化UI元素
        this.menu = document.getElementById('menu');
        this.gameOverScreen = document.getElementById('gameOver');
        this.finalScoreElement = document.getElementById('finalScore');
        this.highScoreElement = document.getElementById('highScore');
        
        // 显示最高分
        this.highScoreElement.textContent = this.highScore;
    }
    
    bindEvents() {
        // 开始游戏
        document.getElementById('startButton').addEventListener('click', () => this.startGame());
        document.getElementById('restartButton').addEventListener('click', () => this.startGame());
        
        // 跳跃控制
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && this.isPlaying) {
                // 如果在地面或者还可以二段跳
                if (this.canJump()) {
                    this.startJump();
                }
            }
        });

        document.addEventListener('keyup', (e) => {
            if (e.code === 'Space') {
                this.player.isJumping = false;
            }
        });
    }
    
    canJump() {
        // 在地面上
        if (this.player.y >= this.canvas.height - this.player.radius * 2) {
            this.player.jumpCount = 0; // 重置跳跃次数
            return true;
        }
        // 在空中且还可以跳
        return this.player.jumpCount < this.player.maxJumps;
    }
    
    startJump() {
        this.player.isJumping = true;
        this.player.jumpStartTime = Date.now();
        
        // 根据跳跃次数决定跳跃力度
        if (this.player.jumpCount === 0) {
            // 第一段跳，使用原有力度
            this.player.velocity = this.player.baseJumpForce;
        } else {
            // 第二段跳，力度减半
            this.player.velocity = this.player.baseJumpForce * this.player.doubleJumpMultiplier;
        }
        
        this.player.jumpCount++;
    }
    
    updatePlayer() {
        // 更新跳跃力度
        if (this.player.isJumping && this.player.velocity < 0) {
            const jumpTime = Date.now() - this.player.jumpStartTime;
            if (jumpTime < this.player.maxJumpTime) {
                const jumpRatio = jumpTime / this.player.maxJumpTime;
                const extraForce = this.player.maxExtraForce * (1 - jumpRatio);
                // 根据跳跃次数调整额外力度
                const multiplier = this.player.jumpCount === 1 ? 1 : this.player.doubleJumpMultiplier;
                this.player.velocity += (extraForce / 10) * multiplier;
            }
        }

        // 应用重力
        this.player.velocity += this.player.gravity;
        this.player.y += this.player.velocity;
        
        // 地面碰撞检测
        if (this.player.y > this.canvas.height - this.player.radius * 2) {
            this.player.y = this.canvas.height - this.player.radius * 2;
            this.player.velocity = 0;
            this.player.jumpCount = 0; // 着地时重置跳跃次数
        }
    }
    
    startGame() {
        this.isPlaying = true;
        this.score = 0;
        this.obstacles = [];
        this.player.y = this.canvas.height - this.player.radius * 2;
        this.player.velocity = 0;
        this.player.jumpCount = 0; // 重置跳跃次数
        this.menu.style.display = 'none';
        this.gameOverScreen.style.display = 'none';
        this.gameLoop();
    }
    
    getRandomInterval() {
        return Math.random() * (this.maxObstacleInterval - this.minObstacleInterval) + this.minObstacleInterval;
    }
    
    createObstacle() {
        const selectedPattern = this.obstacleTypes[Math.floor(Math.random() * this.obstacleTypes.length)];
        const obstacles = [];
        let totalWidth = 0;

        // 创建组合障碍物
        selectedPattern.pattern.forEach((type, index) => {
            const baseType = this.baseObstacleTypes[type];
            const obstacle = {
                x: this.canvas.width + totalWidth,
                y: this.canvas.height - baseType.height,
                width: baseType.width,
                height: baseType.height,
                color: baseType.color
            };
            
            // 如果不是第一个障碍物，设置适当的间距
            if (index > 0) {
                obstacle.x += 10; // 障碍物之间的间距
                totalWidth += 10;
            }
            
            totalWidth += baseType.width;
            obstacles.push(obstacle);
        });

        // 将创建的障碍物添加到游戏中
        this.obstacles.push(...obstacles);
        this.nextObstacleInterval = this.getRandomInterval();
    }
    
    updateObstacles() {
        for (let i = this.obstacles.length - 1; i >= 0; i--) {
            const obstacle = this.obstacles[i];
            obstacle.x -= this.gameSpeed;
            
            // 移除超出画面的障碍物
            if (obstacle.x + obstacle.width < 0) {
                this.obstacles.splice(i, 1);
                // 只在最后一个障碍物完全离开屏幕时增加分数
                if (i === 0 || this.obstacles[i-1].x + this.obstacles[i-1].width > 0) {
                    this.score++;
                }
            }
            
            // 碰撞检测
            if (this.checkCollision(this.player, obstacle)) {
                this.gameOver();
            }
        }
        
        // 生成新障碍物
        if (Date.now() - this.obstacleTimer > this.nextObstacleInterval) {
            this.createObstacle();
            this.obstacleTimer = Date.now();
        }
    }
    
    checkCollision(player, obstacle) {
        // 简化的碰撞检测：使用圆形的边界框与三角形的边界框进行检测
        const circleX = player.x;
        const circleY = player.y + player.radius;
        
        // 检查圆的边界框是否与三角形的边界框重叠
        if (circleX + player.radius < obstacle.x) return false;
        if (circleX - player.radius > obstacle.x + obstacle.width) return false;
        if (circleY + player.radius < obstacle.y) return false;
        if (circleY - player.radius > obstacle.y + obstacle.height) return false;
        
        return true;
    }
    
    draw() {
        // 清空画布
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 绘制地面
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(0, this.canvas.height - 20, this.canvas.width, 20);
        
        // 绘制玩家（圆形）
        this.ctx.beginPath();
        this.ctx.arc(this.player.x, this.player.y + this.player.radius, 
                     this.player.radius, 0, Math.PI * 2);
        
        // 如果正在跳跃，改变颜色
        if (this.player.isJumping && this.player.velocity < 0) {
            const jumpTime = Date.now() - this.player.jumpStartTime;
            const jumpRatio = Math.min(jumpTime / this.player.maxJumpTime, 1);
            this.ctx.fillStyle = this.player.jumpingColor;
        } else {
            this.ctx.fillStyle = this.player.baseColor;
        }
        
        this.ctx.fill();
        this.ctx.closePath();
        
        // 绘制障碍物（三角形）
        this.obstacles.forEach(obstacle => {
            this.ctx.beginPath();
            this.ctx.moveTo(obstacle.x, obstacle.y + obstacle.height);
            this.ctx.lineTo(obstacle.x + obstacle.width/2, obstacle.y);
            this.ctx.lineTo(obstacle.x + obstacle.width, obstacle.y + obstacle.height);
            this.ctx.closePath();
            this.ctx.fillStyle = obstacle.color;
            this.ctx.fill();
        });
        
        // 绘制分数
        this.ctx.fillStyle = '#333';
        this.ctx.font = '20px Arial';
        this.ctx.fillText(`分数: ${this.score}`, 20, 30);
    }
    
    gameOver() {
        this.isPlaying = false;
        if (this.score > this.highScore) {
            this.highScore = this.score;
            localStorage.setItem('highScore', this.highScore);
            this.highScoreElement.textContent = this.highScore;
        }
        this.finalScoreElement.textContent = this.score;
        this.gameOverScreen.style.display = 'block';
    }
    
    gameLoop() {
        if (!this.isPlaying) return;
        
        this.updatePlayer();
        this.updateObstacles();
        this.draw();
        
        requestAnimationFrame(() => this.gameLoop());
    }
}

// 初始化游戏
window.onload = () => {
    new Game();
}; 