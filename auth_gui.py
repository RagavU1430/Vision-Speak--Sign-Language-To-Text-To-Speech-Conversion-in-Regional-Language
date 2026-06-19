# -*- coding: utf-8 -*-
"""
VisionSpeak Authentication & Profile GUI Module
================================================
Provides a modern dark-themed desktop interface for Login, Registration,
Password Reset, and Profile Updates using Python's tkinter library.

Integrates with SupabaseManager for authentication and profile database records.
All network actions run in background threads to keep the UI smooth and responsive.
"""

import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from supabase_client import get_supabase

# ── Color Palette (Matching VisionSpeak Neon HUD) ───────────────────────────
BG_DARK = "#14141E"          # Deep obsidian slate
BG_CARD = "#1C1C28"          # Dark slate card background
TXT_WHITE = "#FFFFFF"        # Primary text
TXT_MUTED = "#8A8A9A"        # Muted gray text
BORDER_COLOR = "#5A2878"     # Neon dark purple
ACCENT_CYAN = "#00C2D0"      # Neon Cyan
ACCENT_PURPLE = "#DC50DC"    # Tech Purple
ACCENT_GREEN = "#64FF64"     # Cyber Lime Green
ACCENT_RED = "#FF5050"       # Warning Red


class AuthApp(tk.Tk):
    """
    Main authentication window wrapper managing views (Login & Register).
    """
    def __init__(self):
        super().__init__()
        self.title("VisionSpeak - Secure Access")
        self.geometry("450x650")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        
        # Center the window
        self.center_window()
        
        # Supabase manager
        self.sb = get_supabase()
        self.sb.connect()
        
        # Authenticated user session and profile data
        self.auth_result = None
        
        # Initialize styles
        self.init_styles()
        
        # Show Login Frame by default
        self.show_login_frame()
        
    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def init_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        
        # Global Styles
        style.configure(".", background=BG_DARK, foreground=TXT_WHITE)
        style.configure("TFrame", background=BG_DARK)
        style.configure("Card.TFrame", background=BG_CARD, borderwidth=1, relief="solid")
        
        # Label Styles
        style.configure("TLabel", background=BG_DARK, foreground=TXT_WHITE, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=BG_CARD, foreground=TXT_WHITE, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG_DARK, foreground=ACCENT_CYAN, font=("Segoe UI Semibold", 18, "bold"))
        style.configure("Sub.TLabel", background=BG_DARK, foreground=TXT_MUTED, font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background=BG_CARD, foreground=ACCENT_CYAN, font=("Segoe UI Semibold", 13, "bold"))
        
        # Entry Style
        style.configure("TEntry", fieldbackground=BG_DARK, foreground=TXT_WHITE, bordercolor=BORDER_COLOR, lightcolor=BORDER_COLOR, darkcolor=BORDER_COLOR)
        
        # Combobox
        style.configure("TCombobox", fieldbackground=BG_DARK, foreground=TXT_WHITE, bordercolor=BORDER_COLOR)
        
        # Button Styles
        style.configure("Primary.TButton", background=ACCENT_CYAN, foreground=BG_DARK, font=("Segoe UI Semibold", 10, "bold"), borderwidth=0)
        style.map("Primary.TButton", background=[("active", ACCENT_GREEN)])
        
        style.configure("Secondary.TButton", background=BG_CARD, foreground=ACCENT_PURPLE, font=("Segoe UI", 9), borderwidth=1, bordercolor=BORDER_COLOR)
        style.map("Secondary.TButton", background=[("active", BG_DARK)], foreground=[("active", TXT_WHITE)])

        style.configure("Link.TButton", background=BG_DARK, foreground=ACCENT_CYAN, font=("Segoe UI", 9, "underline"), borderwidth=0)
        style.map("Link.TButton", background=[("active", BG_DARK)], foreground=[("active", ACCENT_PURPLE)])

    def show_login_frame(self):
        self.clear_frame()
        self.current_frame = LoginFrame(self)
        self.current_frame.pack(fill="both", expand=True)

    def show_register_frame(self):
        self.clear_frame()
        self.current_frame = RegisterFrame(self)
        self.current_frame.pack(fill="both", expand=True)

    def clear_frame(self):
        for widget in self.winfo_children():
            widget.destroy()

    def complete_auth(self, result):
        """Successfully authenticated. Store results and close window."""
        self.auth_result = result
        self.destroy()


class LoginFrame(ttk.Frame):
    """
    Login interface panel.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.init_ui()
        
    def init_ui(self):
        # Top Logo / Title Banner
        title_label = ttk.Label(self, text="VisionSpeak", style="Title.TLabel")
        title_label.pack(pady=(40, 5))
        
        subtitle_label = ttk.Label(self, text="Real-Time Sign Language System", style="Sub.TLabel")
        subtitle_label.pack(pady=(0, 20))
        
        # Card container
        card = ttk.Frame(self, style="Card.TFrame")
        card.pack(pady=10, padx=30, fill="both", expand=True)
        
        card_title = ttk.Label(card, text="SECURE LOGIN", style="CardTitle.TLabel")
        card_title.pack(pady=(20, 15))
        
        # Email Field
        email_label = ttk.Label(card, text="Email Address", style="Card.TLabel")
        email_label.pack(anchor="w", padx=25, pady=(5, 2))
        
        self.email_entry = ttk.Entry(card, width=35)
        self.email_entry.pack(padx=25, pady=(0, 10))
        self.email_entry.focus()
        
        # Password Field
        pass_label = ttk.Label(card, text="Password", style="Card.TLabel")
        pass_label.pack(anchor="w", padx=25, pady=(5, 2))
        
        self.pass_entry = ttk.Entry(card, width=35, show="*")
        self.pass_entry.pack(padx=25, pady=(0, 15))
        
        # Loading / Status Label
        self.status_label = ttk.Label(card, text="", style="Card.TLabel")
        self.status_label.pack(pady=5)
        
        # Login Button
        self.login_btn = ttk.Button(card, text="Login", style="Primary.TButton", command=self.handle_login_threaded)
        self.login_btn.pack(pady=10, padx=25, fill="x")
        
        # Forgot Password Link
        self.forgot_btn = ttk.Button(card, text="Forgot Password?", style="Link.TButton", command=self.handle_forgot_password)
        self.forgot_btn.pack(pady=5)
        
        # Create Account Link
        no_acct_label = ttk.Label(card, text="New to VisionSpeak?", style="Card.TLabel", foreground=TXT_MUTED)
        no_acct_label.pack(pady=(15, 0))
        
        self.create_btn = ttk.Button(card, text="Create Account", style="Link.TButton", command=self.parent.show_register_frame)
        self.create_btn.pack(pady=(0, 15))

        # Handle Enter key press
        self.parent.bind("<Return>", lambda event: self.handle_login_threaded())

    def handle_login_threaded(self):
        email = self.email_entry.get().strip()
        password = self.pass_entry.get().strip()
        
        if not email or not password:
            messagebox.showerror("Error", "Please fill in all fields.")
            return
            
        self.set_loading(True)
        threading.Thread(target=self.perform_login, args=(email, password), daemon=True).start()
        
    def perform_login(self, email, password):
        res = self.parent.sb.login_user(email, password)
        self.parent.after(0, lambda: self.finish_login(res))
        
    def finish_login(self, res):
        self.set_loading(False)
        if res.get("success"):
            self.parent.complete_auth(res)
        else:
            messagebox.showerror("Login Failed", res.get("error", "Invalid email or password."))

    def handle_forgot_password(self):
        email = self.email_entry.get().strip()
        if not email:
            messagebox.showwarning("Reset Password", "Please enter your email address in the email field first.")
            return
            
        confirm = messagebox.askyesno("Reset Password", f"Send a password reset link to {email}?")
        if confirm:
            self.set_loading(True)
            threading.Thread(target=self.perform_reset, args=(email,), daemon=True).start()
            
    def perform_reset(self, email):
        success = self.parent.sb.reset_password_for_email(email)
        self.parent.after(0, lambda: self.finish_reset(success, email))
        
    def finish_reset(self, success, email):
        self.set_loading(False)
        if success:
            messagebox.showinfo("Reset Sent", f"Password reset instructions have been sent to {email}.")
        else:
            messagebox.showerror("Error", "Failed to send reset email. Verify your connection or email format.")

    def set_loading(self, loading):
        if loading:
            self.status_label.config(text="Processing request...", foreground=ACCENT_CYAN)
            self.login_btn.state(["disabled"])
            self.forgot_btn.state(["disabled"])
            self.create_btn.state(["disabled"])
        else:
            self.status_label.config(text="")
            self.login_btn.state(["!disabled"])
            self.forgot_btn.state(["!disabled"])
            self.create_btn.state(["!disabled"])


class RegisterFrame(ttk.Frame):
    """
    Registration interface panel.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.init_ui()
        
    def init_ui(self):
        # Card container
        card = ttk.Frame(self, style="Card.TFrame")
        card.pack(pady=20, padx=25, fill="both", expand=True)
        
        card_title = ttk.Label(card, text="CREATE ACCOUNT", style="CardTitle.TLabel")
        card_title.pack(pady=(15, 10))
        
        # Two-column grid setup for form entries
        form_frame = ttk.Frame(card, style="Card.TFrame")
        form_frame.pack(fill="x", padx=15, pady=5)
        
        # Name
        ttk.Label(form_frame, text="Full Name *", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.name_entry = ttk.Entry(form_frame, width=20)
        self.name_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 6))
        
        # Age
        ttk.Label(form_frame, text="Age *", style="Card.TLabel").grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.age_entry = ttk.Entry(form_frame, width=10)
        self.age_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=(0, 6))
        
        # Email
        ttk.Label(form_frame, text="Email Address *", style="Card.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self.email_entry = ttk.Entry(form_frame, width=35)
        self.email_entry.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 6))
        
        # Phone Number
        ttk.Label(form_frame, text="Phone Number *", style="Card.TLabel").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.phone_entry = ttk.Entry(form_frame, width=20)
        self.phone_entry.grid(row=5, column=0, sticky="ew", padx=5, pady=(0, 6))
        
        # Emergency Contact Number
        ttk.Label(form_frame, text="Emergency Contact *", style="Card.TLabel").grid(row=4, column=1, sticky="w", padx=5, pady=2)
        self.emergency_entry = ttk.Entry(form_frame, width=20)
        self.emergency_entry.grid(row=5, column=1, sticky="ew", padx=5, pady=(0, 6))
        
        # Password
        ttk.Label(form_frame, text="Password *", style="Card.TLabel").grid(row=6, column=0, sticky="w", padx=5, pady=2)
        self.pass_entry = ttk.Entry(form_frame, width=20, show="*")
        self.pass_entry.grid(row=7, column=0, sticky="ew", padx=5, pady=(0, 6))
        
        # Confirm Password
        ttk.Label(form_frame, text="Confirm Password *", style="Card.TLabel").grid(row=6, column=1, sticky="w", padx=5, pady=2)
        self.confirm_entry = ttk.Entry(form_frame, width=20, show="*")
        self.confirm_entry.grid(row=7, column=1, sticky="ew", padx=5, pady=(0, 6))
        
        # Preferred Language
        ttk.Label(form_frame, text="Preferred Language", style="Card.TLabel").grid(row=8, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self.lang_var = tk.StringVar(value="English")
        self.lang_combo = ttk.Combobox(form_frame, textvariable=self.lang_var, values=["English", "Tamil"], state="readonly", width=33)
        self.lang_combo.grid(row=9, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 10))
        
        # Make form columns expand evenly
        form_frame.columnconfigure(0, weight=1)
        form_frame.columnconfigure(1, weight=1)
        
        # Loading / Status Label
        self.status_label = ttk.Label(card, text="", style="Card.TLabel")
        self.status_label.pack(pady=3)
        
        # Register Button
        self.register_btn = ttk.Button(card, text="Create Account", style="Primary.TButton", command=self.handle_register_threaded)
        self.register_btn.pack(pady=8, padx=20, fill="x")
        
        # Back to Login Button
        self.back_btn = ttk.Button(card, text="Back to Login", style="Secondary.TButton", command=self.parent.show_login_frame)
        self.back_btn.pack(pady=5, padx=20, fill="x")

    def handle_register_threaded(self):
        name = self.name_entry.get().strip()
        age_str = self.age_entry.get().strip()
        email = self.email_entry.get().strip()
        phone = self.phone_entry.get().strip()
        emergency = self.emergency_entry.get().strip()
        password = self.pass_entry.get().strip()
        confirm = self.confirm_entry.get().strip()
        language = self.lang_var.get()
        
        # Validations
        if not all([name, age_str, email, phone, emergency, password, confirm]):
            messagebox.showerror("Validation Error", "All fields marked with * are required.")
            return
            
        try:
            age = int(age_str)
            if age <= 0 or age > 120:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Validation Error", "Please enter a valid age (integer).")
            return
            
        if password != confirm:
            messagebox.showerror("Validation Error", "Passwords do not match.")
            return
            
        if len(password) < 6:
            messagebox.showerror("Validation Error", "Password must be at least 6 characters.")
            return
            
        self.set_loading(True)
        threading.Thread(
            target=self.perform_registration,
            args=(email, password, name, age, phone, emergency, language),
            daemon=True
        ).start()
        
    def perform_registration(self, email, password, name, age, phone, emergency, language):
        res = self.parent.sb.sign_up_user(email, password, name, age, phone, emergency, language)
        self.parent.after(0, lambda: self.finish_registration(res))
        
    def finish_registration(self, res):
        self.set_loading(False)
        if res.get("success"):
            if res.get("email_confirm_required"):
                messagebox.showinfo("Registration Successful", "Please check your email to confirm registration before logging in.")
                self.parent.show_login_frame()
            else:
                messagebox.showinfo("Registration Successful", "Your VisionSpeak profile has been created.")
                # Automatically log them in since we got a profile
                self.parent.complete_auth(res)
        else:
            messagebox.showerror("Registration Failed", res.get("error", "Error creating account."))

    def set_loading(self, loading):
        if loading:
            self.status_label.config(text="Registering account...", foreground=ACCENT_CYAN)
            self.register_btn.state(["disabled"])
            self.back_btn.state(["disabled"])
        else:
            self.status_label.config(text="")
            self.register_btn.state(["!disabled"])
            self.back_btn.state(["!disabled"])


class ProfileEditDialog(tk.Toplevel):
    """
    Modal dialog to allow logged-in users to update their profile.
    """
    def __init__(self, parent, auth_user_id, current_profile):
        super().__init__(parent)
        self.parent = parent
        self.auth_user_id = auth_user_id
        self.current_profile = current_profile or {}
        
        self.title("VisionSpeak - Edit Profile")
        self.geometry("400x520")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        self.transient(parent)  # Set as transient dialog
        self.grab_set()         # Block events to main window
        
        self.center_window()
        self.sb = get_supabase()
        self.saved_profile = None
        
        self.init_ui()
        
    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def init_ui(self):
        # Card container
        card = ttk.Frame(self, style="Card.TFrame")
        card.pack(pady=20, padx=20, fill="both", expand=True)
        
        card_title = ttk.Label(card, text="EDIT PROFILE DETAILS", style="CardTitle.TLabel")
        card_title.pack(pady=(15, 10))
        
        form_frame = ttk.Frame(card, style="Card.TFrame")
        form_frame.pack(fill="x", padx=20, pady=5)
        
        # Name
        ttk.Label(form_frame, text="Full Name", style="Card.TLabel").pack(anchor="w", pady=(5, 2))
        self.name_entry = ttk.Entry(form_frame, width=32)
        self.name_entry.pack(fill="x", pady=(0, 10))
        self.name_entry.insert(0, self.current_profile.get("name", ""))
        
        # Age
        ttk.Label(form_frame, text="Age", style="Card.TLabel").pack(anchor="w", pady=(5, 2))
        self.age_entry = ttk.Entry(form_frame, width=32)
        self.age_entry.pack(fill="x", pady=(0, 10))
        self.age_entry.insert(0, str(self.current_profile.get("age", "") or ""))
        
        # Phone Number
        ttk.Label(form_frame, text="Phone Number", style="Card.TLabel").pack(anchor="w", pady=(5, 2))
        self.phone_entry = ttk.Entry(form_frame, width=32)
        self.phone_entry.pack(fill="x", pady=(0, 10))
        self.phone_entry.insert(0, self.current_profile.get("phone_number", ""))
        
        # Emergency Contact Number
        ttk.Label(form_frame, text="Emergency Contact", style="Card.TLabel").pack(anchor="w", pady=(5, 2))
        self.emergency_entry = ttk.Entry(form_frame, width=32)
        self.emergency_entry.pack(fill="x", pady=(0, 10))
        self.emergency_entry.insert(0, self.current_profile.get("emergency_contact", ""))
        
        # Preferred Language
        ttk.Label(form_frame, text="Preferred Language", style="Card.TLabel").pack(anchor="w", pady=(5, 2))
        self.lang_var = tk.StringVar(value=self.current_profile.get("preferred_language", "English"))
        self.lang_combo = ttk.Combobox(form_frame, textvariable=self.lang_var, values=["English", "Tamil"], state="readonly")
        self.lang_combo.pack(fill="x", pady=(0, 15))
        
        # Status Label
        self.status_label = ttk.Label(card, text="", style="Card.TLabel")
        self.status_label.pack(pady=2)
        
        # Buttons Frame
        btn_frame = ttk.Frame(card, style="Card.TFrame")
        btn_frame.pack(fill="x", padx=20, pady=10)
        
        self.save_btn = ttk.Button(btn_frame, text="Save Changes", style="Primary.TButton", command=self.handle_save_threaded)
        self.save_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", style="Secondary.TButton", command=self.destroy)
        self.cancel_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

    def handle_save_threaded(self):
        name = self.name_entry.get().strip()
        age_str = self.age_entry.get().strip()
        phone = self.phone_entry.get().strip()
        emergency = self.emergency_entry.get().strip()
        language = self.lang_var.get()
        
        if not all([name, age_str, phone, emergency]):
            messagebox.showerror("Validation Error", "All fields are required.", parent=self)
            return
            
        try:
            age = int(age_str)
            if age <= 0 or age > 120:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Validation Error", "Please enter a valid age (integer).", parent=self)
            return
            
        self.set_loading(True)
        
        updated_data = {
            "name": name,
            "age": age,
            "phone_number": phone,
            "emergency_contact": emergency,
            "preferred_language": language
        }
        
        threading.Thread(target=self.perform_save, args=(updated_data,), daemon=True).start()
        
    def perform_save(self, updated_data):
        success = self.sb.update_user_profile(self.auth_user_id, updated_data)
        self.parent.after(0, lambda: self.finish_save(success, updated_data))
        
    def finish_save(self, success, updated_data):
        self.set_loading(False)
        if success:
            messagebox.showinfo("Success", "Profile updated successfully.", parent=self)
            self.saved_profile = {**self.current_profile, **updated_data}
            self.destroy()
        else:
            messagebox.showerror("Error", "Failed to update profile. Please verify connection.", parent=self)

    def set_loading(self, loading):
        if loading:
            self.status_label.config(text="Saving profile...", foreground=ACCENT_CYAN)
            self.save_btn.state(["disabled"])
            self.cancel_btn.state(["disabled"])
        else:
            self.status_label.config(text="")
            self.save_btn.state(["!disabled"])
            self.cancel_btn.state(["!disabled"])


# ── Launcher API Functions ──────────────────────────────────────────────────

def run_auth_flow() -> dict | None:
    """
    Main entry point for starting authentication.
    Checks auto-login first; if session exists, automatically logs in.
    Else, displays the Login/Registration GUI.
    Returns:
        dict: Authenticated session + profile details, or None if cancelled/closed.
    """
    sb = get_supabase()
    sb.connect()
    
    # 1. Attempt Auto-Login
    print("[AUTH] Checking for cached local session...")
    auto_res = sb.auto_login()
    if auto_res:
        print(f"[AUTH] Auto-login successful for: {auto_res['profile'].get('email', 'Cached User')}")
        return auto_res
        
    print("[AUTH] No valid cached session found. Launching Auth GUI...")
    # 2. Launch Auth GUI
    app = AuthApp()
    app.mainloop()
    
    return app.auth_result


def edit_user_profile(parent_tk_widget, auth_user_id, current_profile) -> dict | None:
    """
    Launches a modal top-level dialog allowing the user to edit their profile details.
    Returns:
        dict: The updated profile if saved, or None if cancelled.
    """
    dialog = ProfileEditDialog(parent_tk_widget, auth_user_id, current_profile)
    parent_tk_widget.wait_window(dialog)
    return dialog.saved_profile


if __name__ == "__main__":
    # Test script launcher
    print("Testing Auth Flow GUI...")
    res = run_auth_flow()
    print("Authentication flow completed. Result:")
    print(res)
