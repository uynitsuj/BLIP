   def open_grippers(self):
        #self.y.left.open_gripper()
        self.y.right.open_gripper()

    def open_gripper(self, which_arm):
        '''
        FLAG: FIX!
        '''
        arm = self.y.right
        #arm = self.y.left if which_arm == "left" else self.y.right
        arm.open_gripper()


    def close_gripper(self, which_arm):
        '''
        FLAG: FIX!
        '''
        arm = self.y.right
        #arm = self.y.left if which_arm == "left" else self.y.right
        arm.close_gripper()

    def close_grippers(self):
        '''
        FLAG: FIX!
        '''
        #self.y.left.close_gripper()
        self.y.right.close_gripper()

    def slide_grippers(self):
        '''
        FLAG: FIX!
        '''
        # self.y.left.move_gripper(self.GRIP_SLIDE_DIST)
        # self.sync()
        self.y.right.move_gripper(self.GRIP_SLIDE_DIST)
        self.sync()
        time.sleep(self.SLIDE_SLEEP_TIME)


    def slide_gripper(self, which_arm):
        '''
        FLAG: FIX!
        '''
        if which_arm == "left":
            pass
            # self.y.left.move_gripper(self.GRIP_SLIDE_DIST)
            # self.y.left.sync()
        else:
            self.y.right.move_gripper(self.GRIP_SLIDE_DIST)
            self.y.right.sync()
        time.sleep(self.SLIDE_SLEEP_TIME)

    def move_gripper(self, which_arm,dist):
        '''
        FLAG: FIX!
        '''
        if which_arm == "left":
            pass
            # self.y.left.move_gripper(dist)
            # self.y.left.sync()
        else:
            self.y.right.move_gripper(dist)
            self.y.right.sync()
        time.sleep(self.SLIDE_SLEEP_TIME)



# GOOD VERSION
    def open_grippers(self):
        self.y.left.open_gripper()
        self.y.right.open_gripper()

    def open_gripper(self, which_arm):
        '''
        FLAG: FIX!
        '''
        # arm = self.y.right
        arm = self.y.left if which_arm == "left" else self.y.right
        arm.open_gripper()


    def close_gripper(self, which_arm):
        '''
        FLAG: FIX!
        '''
        # arm = self.y.right
        arm = self.y.left if which_arm == "left" else self.y.right
        arm.close_gripper()

    def close_grippers(self):
        self.y.left.close_gripper()
        self.y.right.close_gripper()

    def slide_grippers(self):
        self.y.left.move_gripper(self.GRIP_SLIDE_DIST)
        self.sync()
        self.y.right.move_gripper(self.GRIP_SLIDE_DIST)
        self.sync()
        time.sleep(self.SLIDE_SLEEP_TIME)


    def slide_gripper(self, which_arm):
        '''
        FLAG: FIX!
        '''
        if which_arm == "left":
            self.y.left.move_gripper(self.GRIP_SLIDE_DIST)
            self.y.left.sync()
        else:
            self.y.right.move_gripper(self.GRIP_SLIDE_DIST)
            self.y.right.sync()
        time.sleep(self.SLIDE_SLEEP_TIME)

    def move_gripper(self, which_arm,dist):
        '''
        FLAG: FIX!
        '''
        if which_arm == "left":
            self.y.left.move_gripper(dist)
            self.y.left.sync()
        else:
            self.y.right.move_gripper(dist)
            self.y.right.sync()
        time.sleep(self.SLIDE_SLEEP_TIME)